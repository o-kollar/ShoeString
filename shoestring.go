package main

import (
	"bufio"
	"encoding/gob" // Use gob for BPE state as well
	"errors"
	"flag"
	"fmt"
	"io/ioutil" // Keep for data loading
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	"runtime"
)

// --- Configuration Variables (set by flags with defaults) ---
var (
	flagBPEVocabSize       int
	flagEmbeddingDimension int
	flagGRUHiddenSize      int
	flagGRULayers          int
	flagNumExperts         int
	flagTrainSeqLength     int
	flagBatchSize          int
	flagEpochs             int
	flagMaxResponseLength  int
	flagLearningRate       float64
	flagWeightDecay        float64
	flagEpsilonRMSNorm     float64
	flagEpsilonAdamW       float64
	flagGradientClipValue  float64
	// New flags for separated BPE and mode control
	flagMode        string
	flagBPEPath     string // Path to load existing BPE state
	flagBPEOutputPath string // Path to save trained BPE state
	flagBPEData     string // Path to data for BPE training
	flagModelData   string // Path to data for model training
	flagValData     string // Path to data for validation
	flagCheckpoint  string // Path to load/save model checkpoint
)

// --- Other Constants ---
const (
	CheckpointDir = "checkpoints" // Directory to save model checkpoints
	// NEW: Chunk size for element-wise operations
	defaultChunkSize = 64
)

// Special BPE Tokens
var BpeSpecialTokens = []string{"[USER]", "[BOT]", "[EOS]", "[PAD]", "[UNK]"}

// --- Global Variables ---
var (
	model              map[string]*Mat // Represents the neural network parameters
	solver             *SolverAdamW
	bpe                *BPE // BPE tokenizer instance (loaded or trained)
	batches            [][]TrainingSample // Store training data in batches
	validationBatches  [][]TrainingSample // Store validation data in batches
	trainingComplete   bool               = false // Tracks if model is ready for chat
	bpeActualVocabSize int                // Derived from loaded/trained BPE
	hiddenSizes        []int              // Derived from flags GRULayers and GRUHiddenSize
	seqLength          int                // Corresponds to flagTrainSeqLength
	numExperts         int                // Corresponds to flagNumExperts
	batchSize          int                // Corresponds to flagBatchSize
	embeddingDimension int                // Corresponds to flagEmbeddingDimension
	// Note: LearningRate etc. are used directly via their flag variables or loaded from checkpoint
)

// Training sample structure (Keep as is)
type TrainingSample struct {
	Input  []int
	Target []int
}

// --- Assertion Helper --- (Keep as is)
func assert(condition bool, message string) {
	if !condition {
		log.Panicf("Assertion failed: %s", message)
	}
}

//======================================================================
// *** BPE Tokenizer Implementation ***
//======================================================================
type MergeInfo struct {
	Rank          int      `json:"-"` // No JSON tags needed now
	MergedTokenID int      `json:"-"`
	Pair          []string `json:"-"`
	Result        string   `json:"-"`
	ID            int      `json:"-"`
}

// BPESavedState is now the primary struct for saving/loading BPE state via gob.
type BPESavedState struct {
	SpecialTokens []string
	VocabArray    []string
	Merges        map[string]MergeInfo // Key: "token1 token2"
}

type BPE struct {
	specialTokens    []string
	vocabArray       []string
	vocabMap         map[string]int
	specialTokensMap map[string]int
	merges           map[string]MergeInfo // key: "token1 token2"
	endOfWordSuffix  string
	splitRegex       *regexp.Regexp
	logCallback      func(string) // Optional logging callback
}

func NewBPE(specialTokens []string) *BPE {
	b := &BPE{
		specialTokens:    append([]string{}, specialTokens...), // Copy slice
		vocabArray:       []string{},
		vocabMap:         make(map[string]int),
		specialTokensMap: make(map[string]int),
		merges:           make(map[string]MergeInfo),
		endOfWordSuffix:  "</w>", // Default assumption
		logCallback:      func(msg string) { fmt.Println("BPE:", msg) }, // Default logger
	}
	// Sort special tokens by length descending for regex priority
	sort.Slice(b.specialTokens, func(i, j int) bool {
		return len(b.specialTokens[i]) > len(b.specialTokens[j])
	})
	b.splitRegex = b.buildSplitRegex()
	return b
}

func (b *BPE) escapeRegex(s string) string {
	return regexp.QuoteMeta(s)
}

func (b *BPE) buildSplitRegex() *regexp.Regexp {
	if len(b.specialTokens) == 0 {
		// Matches whitespace sequences or non-whitespace sequences
		return regexp.MustCompile(`(\s+)|([^\s]+)`)
	}
	escapedSpecial := make([]string, len(b.specialTokens))
	for i, token := range b.specialTokens {
		escapedSpecial[i] = b.escapeRegex(token)
	}
	// Pattern: (special_token)|(whitespace)|(non_whitespace_word)
	pattern := fmt.Sprintf(`(%s)|(\s+)|([^\s]+)`, strings.Join(escapedSpecial, "|"))
	return regexp.MustCompile(pattern)
}

func (b *BPE) log(message string) {
	if b.logCallback != nil {
		b.logCallback(message)
	}
}

// Returns slice where elements are either string (special token) or []string (word characters + suffix)
func (b *BPE) preTokenize(text string) []interface{} {
	if text == "" {
		return []interface{}{}
	}
	var segments []interface{}
	matches := b.splitRegex.FindAllStringSubmatch(text, -1)

	for _, match := range matches {
		specialMatch := match[1] // Captured special token group
		// whitespaceMatch := match[2] // Whitespace group (often ignored directly)
		wordMatch := match[3] // Non-whitespace word group

		if specialMatch != "" {
			segments = append(segments, specialMatch)
		} else if wordMatch != "" {
			chars := strings.Split(wordMatch, "")
			wordTokens := append(chars, b.endOfWordSuffix)
			segments = append(segments, wordTokens)
		}
		// We don't explicitly add whitespace segments
	}
	return segments
}

func (b *BPE) getPairStats(corpusWordTokens [][]string) map[string]int {
	stats := make(map[string]int)
	for _, tokens := range corpusWordTokens {
		if len(tokens) < 2 {
			continue
		}
		for i := 0; i < len(tokens)-1; i++ {
			pairKey := fmt.Sprintf("%s %s", tokens[i], tokens[i+1])
			stats[pairKey]++
		}
	}
	return stats
}

func (b *BPE) findBestPair(stats map[string]int) (string, bool) {
	bestPairKey := ""
	maxFreq := -1
	found := false
	for pairKey, freq := range stats {
		if freq > maxFreq {
			maxFreq = freq
			bestPairKey = pairKey
			found = true
		}
	}
	if maxFreq <= 0 { // Ensure we find a pair with frequency > 0
		return "", false
	}
	return bestPairKey, found
}

func (b *BPE) mergeWordTokens(wordTokens []string, pairKey string, mergedToken string) []string {
	pair := strings.Split(pairKey, " ")
	token1 := pair[0]
	token2 := pair[1]
	newWordTokens := []string{}
	i := 0
	for i < len(wordTokens) {
		if i < len(wordTokens)-1 && wordTokens[i] == token1 && wordTokens[i+1] == token2 {
			newWordTokens = append(newWordTokens, mergedToken)
			i += 2
		} else {
			newWordTokens = append(newWordTokens, wordTokens[i])
			i += 1
		}
	}
	return newWordTokens
}

func (b *BPE) addTokenToVocab(token string, isSpecial bool) int {
	if id, exists := b.vocabMap[token]; exists {
		return id
	}
	newID := len(b.vocabArray)
	b.vocabArray = append(b.vocabArray, token)
	b.vocabMap[token] = newID
	if isSpecial {
		b.specialTokensMap[token] = newID
	}
	return newID
}

// Train uses the flagBPEVocabSize variable
func (b *BPE) Train(corpus string, vocabSize int, verbose bool, logCallback func(string)) {
	if logCallback != nil {
		b.logCallback = logCallback
	}
	b.vocabArray = []string{}
	b.vocabMap = make(map[string]int)
	b.specialTokensMap = make(map[string]int)
	b.merges = make(map[string]MergeInfo)

	b.log("--- Starting Tokenizer Training ---")
	b.log(fmt.Sprintf("Target Vocab Size (from flag): %d", vocabSize))
	b.log(fmt.Sprintf("Using special tokens: %s", strings.Join(b.specialTokens, ", ")))

	for _, token := range b.specialTokens {
		b.addTokenToVocab(token, true)
	}
	numSpecialTokens := len(b.specialTokensMap)
	b.log(fmt.Sprintf("Added %d special tokens.", numSpecialTokens))

	uniqueChars := make(map[string]struct{})
	initialSegments := b.preTokenize(corpus)
	var corpusWordTokens [][]string // Store only the word parts for merging

	for _, segment := range initialSegments {
		if wordToks, ok := segment.([]string); ok {
			corpusWordTokens = append(corpusWordTokens, wordToks)
			for _, char := range wordToks {
				if _, isSpecial := b.specialTokensMap[char]; !isSpecial && char != b.endOfWordSuffix { // Add chars if not special
					uniqueChars[char] = struct{}{}
				}
			}
			// Ensure suffix is in vocab if it exists
			if _, isSpecial := b.specialTokensMap[b.endOfWordSuffix]; !isSpecial {
				uniqueChars[b.endOfWordSuffix] = struct{}{}
			}
		}
	}

	// Add unique characters to vocab (sorted for consistency)
	sortedChars := make([]string, 0, len(uniqueChars))
	for char := range uniqueChars {
		sortedChars = append(sortedChars, char)
	}
	sort.Strings(sortedChars)
	for _, char := range sortedChars {
		if _, exists := b.vocabMap[char]; !exists { // Check again in case a char was also a special token
			b.addTokenToVocab(char, false)
		}
	}

	initialVocabSize := len(b.vocabArray)
	b.log(fmt.Sprintf("Initial vocab size (special + chars): %d", initialVocabSize))

	if vocabSize < initialVocabSize {
		b.log(fmt.Sprintf("Warning: Target vocabSize (%d) < initial (%d). No merges.", vocabSize, initialVocabSize))
		vocabSize = initialVocabSize
	}

	numMerges := vocabSize - initialVocabSize
	b.log(fmt.Sprintf("Merges to perform: %d", numMerges))

	if numMerges <= 0 {
		b.log("No merges needed.")
		b.log("--- Tokenizer Training Finished ---")
		b.logCallback = nil // Reset callback
		return
	}

	for i := 0; i < numMerges; i++ {
		stats := b.getPairStats(corpusWordTokens)
		if len(stats) == 0 {
			b.log(fmt.Sprintf("No more pairs at iter %d. Stop.", i+1))
			break
		}

		bestPairKey, found := b.findBestPair(stats)
		if !found {
			b.log(fmt.Sprintf("No beneficial pairs at iter %d. Stop.", i+1))
			break
		}

		pair := strings.Split(bestPairKey, " ")
		token1 := pair[0]
		token2 := pair[1]
		mergedToken := token1 + token2

		if _, isSpecial := b.specialTokensMap[mergedToken]; isSpecial {
			if verbose {
				b.log(fmt.Sprintf("Skip merge %d: Result '%s' is special.", i+1, mergedToken))
			}
			delete(stats, bestPairKey) // Prevent immediate re-selection if possible
			i--                        // Decrement i because this wasn't a real merge step
			continue
		}

		mergedTokenID := b.addTokenToVocab(mergedToken, false)
		b.merges[bestPairKey] = MergeInfo{Rank: i, MergedTokenID: mergedTokenID}

		newCorpusWordTokens := make([][]string, len(corpusWordTokens))
		for j, wordTokens := range corpusWordTokens {
			newCorpusWordTokens[j] = b.mergeWordTokens(wordTokens, bestPairKey, mergedToken)
		}
		corpusWordTokens = newCorpusWordTokens // Update corpus representation

		if verbose && ((i+1)%25 == 0 || i == numMerges-1 || i < 5 || numMerges < 10) {
			freq := stats[bestPairKey] // Get frequency before modifying corpus
			b.log(fmt.Sprintf("Merge %d/%d: '%s' + '%s' -> '%s' (Freq: %d, Vocab: %d)",
				i+1, numMerges, token1, token2, mergedToken, freq, len(b.vocabArray)))
		}

		if len(b.vocabArray) >= vocabSize {
			b.log(fmt.Sprintf("Target size %d reached at merge %d. Stop.", vocabSize, i+1))
			break
		}
	}

	b.log(fmt.Sprintf("Training attempts done. Final vocab size: %d", len(b.vocabArray)))
	b.logCallback = nil // Reset callback
}

// Encode remains the same
func (b *BPE) Encode(text string) []int {
	if len(b.vocabArray) == 0 {
		log.Panic("BPE model not trained or loaded.")
	}

	unkTokenName := "[UNK]"
	unkID, hasUnk := b.specialTokensMap[unkTokenName]
	encodedIDs := []int{}
	segments := b.preTokenize(text)

	for _, segment := range segments {
		if specialToken, ok := segment.(string); ok {
			if id, exists := b.specialTokensMap[specialToken]; exists {
				encodedIDs = append(encodedIDs, id)
			} else {
				if hasUnk {
					encodedIDs = append(encodedIDs, unkID)
					log.Printf("Warning: Unknown special segment '%s' mapped to UNK.", specialToken)
				} else {
					encodedIDs = append(encodedIDs, -1) // Indicate error or unknown without UNK
					log.Printf("Warning: Unknown special segment '%s' and no UNK defined.", specialToken)
				}
			}
		} else if wordChars, ok := segment.([]string); ok {
			wordTokens := append([]string{}, wordChars...) // Copy slice

			// Iteratively apply merges based on rank
			for len(wordTokens) > 1 {
				bestMerge := struct {
					rank    int
					index   int
					pairKey string
				}{rank: math.MaxInt32, index: -1, pairKey: ""}

				for i := 0; i < len(wordTokens)-1; i++ {
					pairKey := fmt.Sprintf("%s %s", wordTokens[i], wordTokens[i+1])
					if mergeInfo, exists := b.merges[pairKey]; exists {
						if mergeInfo.Rank < bestMerge.rank {
							bestMerge.rank = mergeInfo.Rank
							bestMerge.index = i
							bestMerge.pairKey = pairKey
						}
					}
				}

				if bestMerge.rank == math.MaxInt32 {
					break // No more merges possible for this word
				}

				mergedTokenID := b.merges[bestMerge.pairKey].MergedTokenID
				mergedToken := b.vocabArray[mergedTokenID]

				// Replace the pair with the merged token
				newTokenList := []string{}
				newTokenList = append(newTokenList, wordTokens[:bestMerge.index]...)
				newTokenList = append(newTokenList, mergedToken)
				newTokenList = append(newTokenList, wordTokens[bestMerge.index+2:]...)
				wordTokens = newTokenList
			}

			// Convert final tokens to IDs
			for _, token := range wordTokens {
				if id, exists := b.vocabMap[token]; exists {
					encodedIDs = append(encodedIDs, id)
				} else {
					if hasUnk {
						encodedIDs = append(encodedIDs, unkID)
						log.Printf("Warning: Unknown sub-token '%s' mapped to UNK.", token)
					} else {
						encodedIDs = append(encodedIDs, -1)
						log.Printf("Warning: Unknown sub-token '%s' and no UNK defined.", token)
					}
				}
			}
		}
	}
	return encodedIDs
}

// Decode remains the same
func (b *BPE) Decode(tokenIDs []int) string {
	if len(b.vocabArray) == 0 {
		log.Panic("BPE model not trained or loaded.")
	}
	unkTokenName := "[UNK]"
	unkTokenString := "<UNK>" // Default display for unknown ID if UNK token doesn't exist
	if _, hasUnk := b.specialTokensMap[unkTokenName]; hasUnk {
		unkTokenString = unkTokenName
	}

	tokens := make([]string, len(tokenIDs))
	for i, id := range tokenIDs {
		if id >= 0 && id < len(b.vocabArray) {
			tokens[i] = b.vocabArray[id]
		} else {
			tokens[i] = unkTokenString
		}
	}

	decodedText := strings.Join(tokens, "")

	// Replace end-of-word suffix and clean up whitespace
	if b.endOfWordSuffix != "" {
		decodedText = strings.ReplaceAll(decodedText, b.endOfWordSuffix, " ")
	}
	// Consolidate multiple spaces and trim
	spaceRegex := regexp.MustCompile(`\s+`)
	decodedText = spaceRegex.ReplaceAllString(decodedText, " ")
	decodedText = strings.TrimSpace(decodedText)

	return decodedText
}

// GetVocab remains the same
func (b *BPE) GetVocab() []string {
	return append([]string{}, b.vocabArray...) // Return a copy
}

// GetMerges remains the same
func (b *BPE) GetMerges() []MergeInfo {
	mergesList := make([]MergeInfo, 0, len(b.merges))
	for pk, info := range b.merges {
		info.Pair = strings.Split(pk, " ")
		if info.MergedTokenID >= 0 && info.MergedTokenID < len(b.vocabArray) {
			info.Result = b.vocabArray[info.MergedTokenID]
		} else {
			info.Result = "<INVALID>"
		}
		info.ID = info.MergedTokenID // Add ID field for clarity
		mergesList = append(mergesList, info)
	}
	sort.Slice(mergesList, func(i, j int) bool {
		return mergesList[i].Rank < mergesList[j].Rank
	})
	return mergesList
}

// GetState remains the same
func (b *BPE) GetState() BPESavedState {
	// Create copies to avoid modifying original maps/slices if state is held
	vocabCopy := make([]string, len(b.vocabArray))
	copy(vocabCopy, b.vocabArray)
	mergesCopy := make(map[string]MergeInfo, len(b.merges))
	for k, v := range b.merges {
		mergesCopy[k] = v // MergeInfo is simple struct, shallow copy ok
	}
	specialCopy := make([]string, len(b.specialTokens))
	copy(specialCopy, b.specialTokens)

	return BPESavedState{
		SpecialTokens: specialCopy,
		VocabArray:    vocabCopy,
		Merges:        mergesCopy,
	}
}

// LoadState remains the same
func (b *BPE) LoadState(state BPESavedState) error {
	if state.VocabArray == nil || state.Merges == nil || state.SpecialTokens == nil {
		return errors.New("invalid BPE saved state: missing fields")
	}

	// Take ownership of the loaded state's slices/maps
	b.specialTokens = state.SpecialTokens
	b.vocabArray = state.VocabArray
	b.merges = state.Merges

	// Rebuild internal maps from loaded arrays
	b.vocabMap = make(map[string]int, len(b.vocabArray))
	for id, token := range b.vocabArray {
		b.vocabMap[token] = id
	}
	b.specialTokensMap = make(map[string]int)
	for _, token := range b.specialTokens {
		if id, exists := b.vocabMap[token]; exists {
			b.specialTokensMap[token] = id
		} else {
			log.Printf("Warning: Loaded special token '%s' not in vocab.", token)
		}
	}

	// Infer endOfWordSuffix (simple check)
	b.endOfWordSuffix = ""
	for _, token := range b.vocabArray {
		if strings.HasSuffix(token, "</w>") && len(token) > 4 { // Be specific
			b.endOfWordSuffix = "</w>"
			break
		}
	}
	if b.endOfWordSuffix == "" {
		if _, exists := b.vocabMap["</w>"]; exists {
			b.endOfWordSuffix = "</w>"
		}
	}

	// Rebuild regex based on loaded special tokens
	sort.Slice(b.specialTokens, func(i, j int) bool { // Ensure sort order
		return len(b.specialTokens[i]) > len(b.specialTokens[j])
	})
	b.splitRegex = b.buildSplitRegex()

	b.log("BPE state loaded.")
	return nil
}

//======================================================================
// *** BPE State Persistence (NEW FUNCTIONS) ***
//======================================================================

// SaveBPEState saves the BPE state to a file using gob.
func SaveBPEState(bpeInstance *BPE, path string) error {
	if bpeInstance == nil {
		return errors.New("cannot save nil BPE instance")
	}
	log.Printf("Saving BPE state to %s...", path)

	// Ensure the directory exists
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory for BPE file %s: %w", dir, err)
	}

	state := bpeInstance.GetState() // Get the serializable state

	tempPath := path + ".tmp"
	file, err := os.Create(tempPath)
	if err != nil {
		return fmt.Errorf("failed to create temporary BPE file %s: %w", tempPath, err)
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	err = encoder.Encode(state)
	if err != nil {
		file.Close()
		_ = os.Remove(tempPath)
		return fmt.Errorf("failed to encode BPE state to %s: %w", tempPath, err)
	}

	if err := file.Close(); err != nil {
		_ = os.Remove(tempPath)
		return fmt.Errorf("failed to close temporary BPE file %s before rename: %w", tempPath, err)
	}

	err = os.Rename(tempPath, path)
	if err != nil {
		_ = os.Remove(tempPath)
		return fmt.Errorf("failed to rename temporary BPE file to %s: %w", path, err)
	}

	log.Printf("BPE state saved successfully to %s (Vocab size: %d)", path, len(state.VocabArray))
	return nil
}

// LoadBPEState loads the BPE state from a file using gob.
func LoadBPEState(path string) (*BPE, error) {
	log.Printf("Loading BPE state from %s...", path)

	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open BPE state file %s: %w", path, err)
	}
	defer file.Close()

	var state BPESavedState
	decoder := gob.NewDecoder(file)
	err = decoder.Decode(&state)
	if err != nil {
		return nil, fmt.Errorf("failed to decode gob BPE state data from %s: %w", path, err)
	}

	// Create a new BPE instance and load the state into it
	// Pass the loaded special tokens to NewBPE, LoadState will handle the rest
	newBpe := NewBPE(state.SpecialTokens)
	err = newBpe.LoadState(state)
	if err != nil {
		return nil, fmt.Errorf("failed to load BPE state into new BPE instance: %w", err)
	}

	log.Printf("BPE state loaded successfully from %s (Vocab size: %d)", path, len(newBpe.vocabArray))
	return newBpe, nil
}


//======================================================================
// *** R Library (Matrix Ops, Graph, Activations, RMSNorm, etc.) ***
//======================================================================
// --- Matrix Definition & Methods ---
type Mat struct {
	N  int
	D  int
	W  []float64
	Dw []float64
}
func NewMat(n, d int) *Mat { assert(n >= 0 && d >= 0, "Matrix dimensions must be non-negative"); if n*d == 0 { return &Mat{N: n, D: d, W: []float64{}, Dw: []float64{}} }; w := make([]float64, n*d); dw := make([]float64, n*d); return &Mat{N: n, D: d, W: w, Dw: dw} }
func NewRandMat(n, d int, mu, stddev float64) *Mat { m := NewMat(n, d); for i := range m.W { m.W[i] = rand.NormFloat64()*stddev + mu }; return m }
func Zeros(n int) []float64 { if n <= 0 { return []float64{} }; return make([]float64, n) }
func (m *Mat) Get(row, col int) float64 { assert(row >= 0 && row < m.N && col >= 0 && col < m.D, fmt.Sprintf("Mat.Get index (%d,%d) out of bounds for %dx%d matrix", row, col, m.N, m.D)); ix := row*m.D + col; return m.W[ix] }
func (m *Mat) Set(row, col int, v float64) { assert(row >= 0 && row < m.N && col >= 0 && col < m.D, fmt.Sprintf("Mat.Set index (%d,%d) out of bounds for %dx%d matrix", row, col, m.N, m.D)); ix := row*m.D + col; m.W[ix] = v }
func (m *Mat) GetCol(col int) *Mat { assert(col >= 0 && col < m.D, fmt.Sprintf("Mat.GetCol index %d out of bounds for %dx%d matrix", col, m.N, m.D)); colMat := NewMat(m.N, 1); for i := 0; i < m.N; i++ { colMat.W[i] = m.Get(i, col) }; return colMat }
func (m *Mat) Clone() *Mat { newM := NewMat(m.N, m.D); copy(newM.W, m.W); return newM } // Could use chunking for large matrices

// --- UPDATED: ZeroGrads with Chunking ---
func (m *Mat) ZeroGrads() {
	nTotal := len(m.Dw)
	dw := m.Dw // Local slice for potential compiler optimization
	for i := 0; i < nTotal; i += defaultChunkSize {
		end := i + defaultChunkSize
		if end > nTotal {
			end = nTotal
		}
		// Iterate over the chunk
		for j := i; j < end; j++ {
			dw[j] = 0.0
		}
	}
}

// --- Graph Definition & Methods ---
type Graph struct { NeedsBackprop bool; Backprop []func(); mu sync.Mutex }
func NewGraph(needsBackprop bool) *Graph { return &Graph{ NeedsBackprop: needsBackprop, Backprop: []func(){}, } }
func (g *Graph) Backward() { for i := len(g.Backprop) - 1; i >= 0; i-- { g.Backprop[i]() } }
func (g *Graph) addBackward(f func()) { if g.NeedsBackprop { g.mu.Lock(); g.Backprop = append(g.Backprop, f); g.mu.Unlock() } }

// --- UPDATED: applyActivation with Chunking ---
const ( invSqrt2 = 0.7071067811865476; invSqrt2pi = 0.3989422804014327 )
func applyActivation(g *Graph, m *Mat, activationFn func(float64) float64, derivativeFn func(float64, float64) float64) *Mat {
	out := NewMat(m.N, m.D)
	nTotal := len(m.W)
	mW := m.W
	outW := out.W

	// Forward computation with chunking
	for i := 0; i < nTotal; i += defaultChunkSize {
		end := i + defaultChunkSize
		if end > nTotal {
			end = nTotal
		}
		for j := i; j < end; j++ {
			outW[j] = activationFn(mW[j])
		}
	}

	if g.NeedsBackprop {
		mDw := m.Dw
		outDw := out.Dw
		backward := func() {
			// Backward computation with chunking
			for i := 0; i < nTotal; i += defaultChunkSize {
				end := i + defaultChunkSize
				if end > nTotal {
					end = nTotal
				}
				for j := i; j < end; j++ {
					// Read necessary values once per iteration if possible
					mVal := mW[j]
					outVal := outW[j]
					outGrad := outDw[j]
					// Apply derivative and accumulate gradient
					deriv := derivativeFn(mVal, outVal)
					if !math.IsNaN(outGrad) && !math.IsInf(outGrad, 0) && !math.IsNaN(deriv) && !math.IsInf(deriv, 0) {
						mDw[j] += deriv * outGrad
					}
				}
			}
		}
		g.addBackward(backward)
	}
	return out
}

// --- Activation Functions (Benefit from chunked applyActivation) ---
func (g *Graph) Tanh(m *Mat) *Mat { return applyActivation(g, m, math.Tanh, func(m_wi, out_wi float64) float64 { return 1.0 - out_wi*out_wi }) }
func (g *Graph) Sigmoid(m *Mat) *Mat { sigmoid := func(x float64) float64 { return 1.0 / (1.0 + math.Exp(-x)) }; derivative := func(m_wi, out_wi float64) float64 { return out_wi * (1.0 - out_wi) }; return applyActivation(g, m, sigmoid, derivative) }
func (g *Graph) Relu(m *Mat) *Mat { relu := func(x float64) float64 { return math.Max(0, x) }; derivative := func(m_wi, out_wi float64) float64 { if m_wi > 0 { return 1.0 }; return 0.0 }; return applyActivation(g, m, relu, derivative) }
func (g *Graph) Gelu(m *Mat) *Mat { geluFunc := func(x float64) float64 { return 0.5 * x * (1.0 + math.Erf(x*invSqrt2)) }; geluDerivative := func(x, gelu_x float64) float64 { phi_x := invSqrt2pi * math.Exp(-0.5*x*x); var phi_cap_x float64; if math.Abs(x) < 1e-9 { phi_cap_x = 0.5 } else { phi_cap_x = gelu_x / x }; derivative := phi_cap_x + x*phi_x; return derivative }; return applyActivation(g, m, geluFunc, geluDerivative) }

// --- UPDATED: Add with Chunking ---
func (g *Graph) Add(m1, m2 *Mat) *Mat {
	assert(m1.N == m2.N && m1.D == m2.D, fmt.Sprintf("Add: Matrices must have the same size. m1: %dx%d, m2: %dx%d", m1.N, m1.D, m2.N, m2.D))
	out := NewMat(m1.N, m1.D)
	nTotal := len(m1.W)
	w1 := m1.W
	w2 := m2.W
	outW := out.W

	// Forward computation with chunking
	for i := 0; i < nTotal; i += defaultChunkSize {
		end := i + defaultChunkSize
		if end > nTotal {
			end = nTotal
		}
		for j := i; j < end; j++ {
			outW[j] = w1[j] + w2[j]
		}
	}

	if g.NeedsBackprop {
		dw1 := m1.Dw
		dw2 := m2.Dw
		outDw := out.Dw
		backward := func() {
			// Backward computation with chunking
			for i := 0; i < nTotal; i += defaultChunkSize {
				end := i + defaultChunkSize
				if end > nTotal {
					end = nTotal
				}
				for j := i; j < end; j++ {
					grad := outDw[j]
					if !math.IsNaN(grad) && !math.IsInf(grad, 0) { // Added safety check
						dw1[j] += grad
						dw2[j] += grad
					}
				}
			}
		}
		g.addBackward(backward)
	}
	return out
}

// --- Mul (No Chunking Added - complex access pattern) ---
func (g *Graph) Mul(m1, m2 *Mat) *Mat {
	assert(m1.D == m2.N, fmt.Sprintf("Mul: Matrix dimensions misaligned. m1: %dx%d, m2: %dx%d", m1.N, m1.D, m2.N, m2.D))
	n := m1.N; k := m1.D; batchSizeOut := m2.D
	out := NewMat(n, batchSizeOut)
	m1W := m1.W; m2W := m2.W; outW := out.W
	for i := 0; i < n; i++ {
		m1RowOffset := i * k
		for j := 0; j < batchSizeOut; j++ {
			outIdx := i*batchSizeOut + j
			dot := 0.0
			for l := 0; l < k; l++ {
				dot += m1W[m1RowOffset+l] * m2W[l*batchSizeOut+j]
			}
			outW[outIdx] = dot
		}
	}
	if g.NeedsBackprop {
		m1Dw := m1.Dw; m2Dw := m2.Dw; outDw := out.Dw
		backward := func() {
			for i := 0; i < n; i++ {
				m1RowOffset := i * k
				for j := 0; j < batchSizeOut; j++ {
					outIdx := i*batchSizeOut + j
					gradOut := outDw[outIdx]
					if gradOut == 0 { continue }
					for l := 0; l < k; l++ {
						m1Val := m1W[m1RowOffset+l]
						m2Val := m2W[l*batchSizeOut+j]
						m1Dw_ikl := m2Val * gradOut
						m2Dw_lj := m1Val * gradOut
						if !math.IsNaN(m1Dw_ikl) && !math.IsInf(m1Dw_ikl, 0) {
							m1Dw[m1RowOffset+l] += m1Dw_ikl
						}
						if !math.IsNaN(m2Dw_lj) && !math.IsInf(m2Dw_lj, 0) {
							m2Dw[l*batchSizeOut+j] += m2Dw_lj
						}
					}
				}
			}
		}
		g.addBackward(backward)
	}
	return out
}

// --- UPDATED: Eltmul with Chunking ---
func (g *Graph) Eltmul(m1, m2 *Mat) *Mat {
	assert(m1.N == m2.N && m1.D == m2.D, fmt.Sprintf("Eltmul: Matrices must have the same size. m1: %dx%d, m2: %dx%d", m1.N, m1.D, m2.N, m2.D))
	out := NewMat(m1.N, m1.D)
	nTotal := len(m1.W)
	w1 := m1.W
	w2 := m2.W
	outW := out.W

	// Forward computation with chunking
	for i := 0; i < nTotal; i += defaultChunkSize {
		end := i + defaultChunkSize
		if end > nTotal {
			end = nTotal
		}
		for j := i; j < end; j++ {
			outW[j] = w1[j] * w2[j]
		}
	}

	if g.NeedsBackprop {
		dw1 := m1.Dw
		dw2 := m2.Dw
		outDw := out.Dw
		backward := func() {
			// Backward computation with chunking
			for i := 0; i < nTotal; i += defaultChunkSize {
				end := i + defaultChunkSize
				if end > nTotal {
					end = nTotal
				}
				for j := i; j < end; j++ {
					grad := outDw[j]
					w1Val := w1[j]
					w2Val := w2[j]
					if !math.IsNaN(grad) && !math.IsInf(grad, 0) { // Added safety check
						if !math.IsNaN(w2Val) && !math.IsInf(w2Val, 0) {
							dw1[j] += w2Val * grad
						}
						if !math.IsNaN(w1Val) && !math.IsInf(w1Val, 0) {
							dw2[j] += w1Val * grad
						}
					}
				}
			}
		}
		g.addBackward(backward)
	}
	return out
}


// --- AddBroadcastCol (Chunking might be less effective due to column vector broadcast) ---
func (g *Graph) AddBroadcastCol(m1 *Mat, m2Col *Mat) *Mat {
	assert(m1.N == m2Col.N, fmt.Sprintf("AddBroadcastCol: Row dimension mismatch. m1: %dx%d, m2Col: %dx%d", m1.N, m1.D, m2Col.N, m2Col.D))
	assert(m2Col.D == 1, fmt.Sprintf("AddBroadcastCol: m2Col must be a column vector (D=1), got %dx%d", m2Col.N, m2Col.D))
	n := m1.N; batchSize := m1.D
	out := NewMat(n, batchSize)
	m1W := m1.W; m2ColW := m2Col.W; outW := out.W
	for j := 0; j < batchSize; j++ { // Iterate columns (batch)
		m1ColOffset := j
		outColOffset := j
		for i := 0; i < n; i++ { // Iterate rows
			outW[i*batchSize+outColOffset] = m1W[i*batchSize+m1ColOffset] + m2ColW[i]
		}
	}
	if g.NeedsBackprop {
		m1Dw := m1.Dw; m2ColDw := m2Col.Dw; outDw := out.Dw
		backward := func() {
			m2ColDwTemp := Zeros(n) // Temporary accumulator for m2Col gradient
			for j := 0; j < batchSize; j++ { // Iterate columns (batch)
				outColOffset := j
				m1ColOffset := j
				for i := 0; i < n; i++ { // Iterate rows
					gradOut := outDw[i*batchSize+outColOffset]
					if !math.IsNaN(gradOut) && !math.IsInf(gradOut, 0) {
						m1Dw[i*batchSize+m1ColOffset] += gradOut
						m2ColDwTemp[i] += gradOut // Accumulate broadcasts
					}
				}
			}
			// Apply accumulated gradient to m2Col.Dw
			for i := 0; i < n; i++ {
				m2ColDw[i] += m2ColDwTemp[i]
			}
		}
		g.addBackward(backward)
	}
	return out
}

// --- Ones (Chunking applicable but potentially overkill for simple init) ---
func (g *Graph) Ones(n, d int) *Mat {
	m := NewMat(n, d)
	nTotal := len(m.W)
	w := m.W
	// Forward computation with chunking (simple assignment)
	for i := 0; i < nTotal; i += defaultChunkSize {
		end := i + defaultChunkSize
		if end > nTotal {
			end = nTotal
		}
		for j := i; j < end; j++ {
			w[j] = 1.0
		}
	}
	return m // No backprop needed for Ones itself
}

// --- UPDATED: OneMinus with Chunking ---
func (g *Graph) OneMinus(m *Mat) *Mat {
	out := NewMat(m.N, m.D)
	nTotal := len(m.W)
	mW := m.W
	outW := out.W

	// Forward computation with chunking
	for i := 0; i < nTotal; i += defaultChunkSize {
		end := i + defaultChunkSize
		if end > nTotal {
			end = nTotal
		}
		for j := i; j < end; j++ {
			outW[j] = 1.0 - mW[j]
		}
	}

	if g.NeedsBackprop {
		mDw := m.Dw
		outDw := out.Dw
		backward := func() {
			// Backward computation with chunking
			for i := 0; i < nTotal; i += defaultChunkSize {
				end := i + defaultChunkSize
				if end > nTotal {
					end = nTotal
				}
				for j := i; j < end; j++ {
					grad := outDw[j]
					if !math.IsNaN(grad) && !math.IsInf(grad, 0) { // Added safety check
						mDw[j] += -1.0 * grad
					}
				}
			}
		}
		g.addBackward(backward)
	}
	return out
}

// --- Lookup (Chunking less applicable - copies specific rows based on IDs) ---
func (g *Graph) Lookup(embeddingMatrix *Mat, tokenIDs []int) *Mat {
	vocabSize := embeddingMatrix.N; embeddingDim := embeddingMatrix.D
	batchSize := len(tokenIDs)
	assert(batchSize > 0, "Lookup: tokenIDs slice cannot be empty.")
	out := NewMat(embeddingDim, batchSize)
	validIndices := make([]int, batchSize) // Keep track of which indices were valid
	embeddingW := embeddingMatrix.W; outW := out.W

	for j, tokenID := range tokenIDs {
		validIndices[j] = tokenID // Store original ID for backprop mapping
		if tokenID < 0 || tokenID >= vocabSize {
			validIndices[j] = -1 // Mark as invalid for backprop
			// Optionally zero out the corresponding output column if needed, or leave as is (zeros)
			continue
		}
		// Copy the embedding vector
		srcOffset := tokenID * embeddingDim
		destColOffset := j
		for i := 0; i < embeddingDim; i++ {
			outW[i*batchSize+destColOffset] = embeddingW[srcOffset+i]
		}
	}

	if g.NeedsBackprop {
		embeddingDw := embeddingMatrix.Dw; outDw := out.Dw
		backward := func() {
			for j := 0; j < batchSize; j++ { // Iterate through the batch items
				tokenID := validIndices[j]
				if tokenID == -1 { // Skip if the original tokenID was invalid
					continue
				}
				// Add the output gradient back to the corresponding embedding row
				targetRowOffset := tokenID * embeddingDim
				srcColOffset := j
				for i := 0; i < embeddingDim; i++ { // Iterate through embedding dimension
					grad := outDw[i*batchSize+srcColOffset]
					// Add safety check before accumulating gradient
					if !math.IsNaN(grad) && !math.IsInf(grad, 0) {
						embeddingDw[targetRowOffset+i] += grad
					}
				}
			}
		}
		g.addBackward(backward)
	}
	return out
}

// --- CombineExperts (Chunking potentially useful for the inner hiddenSize loops) ---
func (g *Graph) CombineExperts(expertOutputs []*Mat, gatingWeights *Mat) *Mat {
	if len(expertOutputs) == 0 { log.Panic("CombineExperts: expertOutputs slice cannot be empty.") }
	if gatingWeights == nil { log.Panic("CombineExperts: gatingWeights cannot be nil.") }
	numExperts := len(expertOutputs)
	hiddenSize := expertOutputs[0].N; batchSize := expertOutputs[0].D

	// Assertions
	assert(gatingWeights.N == numExperts, fmt.Sprintf("CombineExperts: gatingWeights rows (%d) must match numExperts (%d)", gatingWeights.N, numExperts))
	assert(gatingWeights.D == batchSize, fmt.Sprintf("CombineExperts: gatingWeights cols (%d) must match batch size (%d)", gatingWeights.D, batchSize))
	for e := 0; e < numExperts; e++ {
		assert(expertOutputs[e] != nil, fmt.Sprintf("CombineExperts: expertOutput %d is nil", e))
		assert(expertOutputs[e].N == hiddenSize, fmt.Sprintf("CombineExperts: expertOutput %d rows (%d) must match hiddenSize (%d)", e, expertOutputs[e].N, hiddenSize))
		assert(expertOutputs[e].D == batchSize, fmt.Sprintf("CombineExperts: expertOutput %d cols (%d) must match batch size (%d)", e, expertOutputs[e].D, batchSize))
	}

	out := NewMat(hiddenSize, batchSize)
	outW := out.W; gatingW := gatingWeights.W

	// Forward Pass: Weighted sum of expert outputs
	for e := 0; e < numExperts; e++ {
		expertOut_eW := expertOutputs[e].W
		for j := 0; j < batchSize; j++ { // Iterate batch items
			gateWeight_ej := gatingW[e*batchSize+j]
			if gateWeight_ej == 0 { continue } // Skip if gate weight is zero

			outColOffset := j
			expertColOffset := j

			// --- Potential Chunking Start (Forward Inner Loop) ---
			for i := 0; i < hiddenSize; i += defaultChunkSize {
				end_i := i + defaultChunkSize
				if end_i > hiddenSize { end_i = hiddenSize }
				for row := i; row < end_i; row++ {
					outW[row*batchSize+outColOffset] += expertOut_eW[row*batchSize+expertColOffset] * gateWeight_ej
				}
			}
			// --- Potential Chunking End (Forward Inner Loop) ---
		}
	}

	if g.NeedsBackprop {
		gatingDw := gatingWeights.Dw; outDw := out.Dw
		backward := func() {
			// Need temporary storage for gradients to avoid race conditions if parallelized later,
			// and to correctly accumulate gradients before applying them.
			gatingDwTemp := make([][]float64, numExperts) // [expert][batch]
			for e := range gatingDwTemp { gatingDwTemp[e] = Zeros(batchSize) }
			expertDwTemps := make([][]float64, numExperts) // [expert][flat_hidden*batch]
			for e := range expertDwTemps { expertDwTemps[e] = Zeros(hiddenSize * batchSize) }

			// Calculate gradients w.r.t gating weights and expert outputs
			for e := 0; e < numExperts; e++ {
				expertOut_eW := expertOutputs[e].W
				for j := 0; j < batchSize; j++ { // Iterate batch items
					gradAccumGating_ej := 0.0
					gateWeight_ej := gatingW[e*batchSize+j] // Get the gate weight used in forward

					outColOffset := j
					expertColOffset := j

					// --- Potential Chunking Start (Backward Inner Loops) ---
					for i := 0; i < hiddenSize; i += defaultChunkSize {
						end_i := i + defaultChunkSize
						if end_i > hiddenSize { end_i = hiddenSize }
						for row := i; row < end_i; row++ {
							gradOut_ij := outDw[row*batchSize+outColOffset] // Gradient from downstream

							if !math.IsNaN(gradOut_ij) && !math.IsInf(gradOut_ij, 0) {
								// Grad w.r.t gating weight (dL/dg = dL/dout * dout/dg = dL/dout * expert_out)
								expertVal_eij := expertOut_eW[row*batchSize+expertColOffset]
								gradAccumGating_ej += gradOut_ij * expertVal_eij

								// Grad w.r.t expert output (dL/dexp = dL/dout * dout/dexp = dL/dout * gate_weight)
								gradExpOut_eij := gradOut_ij * gateWeight_ej
								if !math.IsNaN(gradExpOut_eij) && !math.IsInf(gradExpOut_eij, 0) {
									expertDwTemps[e][row*batchSize+expertColOffset] += gradExpOut_eij
								}
							}
						}
					}
					// --- Potential Chunking End (Backward Inner Loops) ---

					// Store accumulated grad for gating weight
					if !math.IsNaN(gradAccumGating_ej) && !math.IsInf(gradAccumGating_ej, 0) {
						gatingDwTemp[e][j] += gradAccumGating_ej
					}
				}
			}

			// Apply accumulated gradients
			for e := 0; e < numExperts; e++ {
				// Apply to expert output gradients
				expertOutDw_e := expertOutputs[e].Dw
				expertDwTemp_e := expertDwTemps[e]
				// --- Potential Chunking Start (Apply Expert Grad) ---
				for idx := 0; idx < len(expertOutDw_e); idx += defaultChunkSize {
					end_idx := idx + defaultChunkSize
					if end_idx > len(expertOutDw_e) { end_idx = len(expertOutDw_e) }
					for k := idx; k < end_idx; k++ {
						expertOutDw_e[k] += expertDwTemp_e[k]
					}
				}
				// --- Potential Chunking End (Apply Expert Grad) ---

				// Apply to gating weight gradients
				gatingDwTemp_e := gatingDwTemp[e]
				// --- Potential Chunking Start (Apply Gating Grad) ---
				for j := 0; j < batchSize; j += defaultChunkSize {
					end_j := j + defaultChunkSize
					if end_j > batchSize { end_j = batchSize }
					for k := j; k < end_j; k++ {
						gatingDw[e*batchSize+k] += gatingDwTemp_e[k]
					}
				}
				// --- Potential Chunking End (Apply Gating Grad) ---
			}
		}
		g.addBackward(backward)
	}
	return out
}


// --- RMSNorm (Chunking potentially useful for the inner hiddenSize loops) ---
func (g *Graph) RMSNorm(m, gain *Mat) *Mat {
	assert(gain.N == m.N, fmt.Sprintf("RMSNorm gain rows must match input rows. m: %dx%d, gain: %dx%d", m.N, m.D, gain.N, gain.D))
	assert(gain.D == 1, fmt.Sprintf("RMSNorm gain must be a column vector (D=1). Got %dx%d", gain.N, gain.D))
	n := m.N; batchSize := m.D
	out := NewMat(n, batchSize)
	rmsPerCol := make([]float64, batchSize) // Stores 1 / sqrt(mean(m^2) + eps) for each column
	invRMSPerCol := make([]float64, batchSize)
	mNorm := NewMat(n, batchSize) // Stores normalized m values (m / rms) for backprop
	mW := m.W; gainW := gain.W; mNormW := mNorm.W; outW := out.W

	// Forward Pass
	for j := 0; j < batchSize; j++ { // Iterate through columns (batch items)
		meanSq := 0.0
		colOffset := j
		// --- Potential Chunking Start (Forward MeanSq) ---
		for i := 0; i < n; i += defaultChunkSize {
			end_i := i + defaultChunkSize
			if end_i > n { end_i = n }
			sumSqChunk := 0.0
			for row := i; row < end_i; row++ {
				val := mW[row*batchSize+colOffset]
				sumSqChunk += val * val
			}
			meanSq += sumSqChunk
		}
		// --- Potential Chunking End (Forward MeanSq) ---
		meanSq /= float64(n)
		rmsPerCol[j] = math.Sqrt(meanSq + flagEpsilonRMSNorm)
		invRMS := 1.0 / rmsPerCol[j]
		invRMSPerCol[j] = invRMS // Store for backprop

		// --- Potential Chunking Start (Forward Normalize & Scale) ---
		for i := 0; i < n; i += defaultChunkSize {
			end_i := i + defaultChunkSize
			if end_i > n { end_i = n }
			for row := i; row < end_i; row++ {
				normVal := mW[row*batchSize+colOffset] * invRMS
				mNormW[row*batchSize+colOffset] = normVal         // Store normalized value
				outW[row*batchSize+colOffset] = normVal * gainW[row] // Apply gain
			}
		}
		// --- Potential Chunking End (Forward Normalize & Scale) ---
	}

	if g.NeedsBackprop {
		mDw := m.Dw; gainDw := gain.Dw; outDw := out.Dw
		backward := func() {
			gainDwTemp := Zeros(n) // Accumulator for gain gradients across batch
			for j := 0; j < batchSize; j++ { // Iterate through columns (batch items)
				sumDNormMTimesNegNormM_j := 0.0
				dNormM_j := Zeros(n) // Grad w.r.t normalized m for this column
				colOffset := j
				invRMS := invRMSPerCol[j]

				// --- Potential Chunking Start (Backward Grad Accumulation) ---
				for i := 0; i < n; i += defaultChunkSize {
					end_i := i + defaultChunkSize
					if end_i > n { end_i = n }
					sumDNormMTimesNegNormM_chunk := 0.0
					for row := i; row < end_i; row++ {
						dOut_ij := outDw[row*batchSize+colOffset] // Gradient from downstream
						if !math.IsNaN(dOut_ij) && !math.IsInf(dOut_ij, 0) {
							mNorm_ij := mNormW[row*batchSize+colOffset] // Normalized value from forward
							gain_i := gainW[row]                      // Gain value

							// Accumulate gradient for gain (dL/dgain = dL/dout * dout/dgain = dL/dout * m_norm)
							gainDwTemp[row] += dOut_ij * mNorm_ij

							// Calculate gradient w.r.t. normalized m (dL/d(m_norm) = dL/dout * dout/d(m_norm) = dL/dout * gain)
							dNormM_ij := dOut_ij * gain_i
							dNormM_j[row] = dNormM_ij

							// Part of the derivative w.r.t RMS (dL/dRMS = sum(dL/d(m_norm) * d(m_norm)/dRMS)))
							// d(m_norm)/dRMS = d(m*invRMS)/dRMS = m * (-invRMS^2) = -m_norm * invRMS
							sumDNormMTimesNegNormM_chunk += dNormM_ij * (-mNorm_ij)
						}
					}
					sumDNormMTimesNegNormM_j += sumDNormMTimesNegNormM_chunk
				}
				// --- Potential Chunking End (Backward Grad Accumulation) ---

				// Calculate gradient w.r.t RMS
				dRMS_j := sumDNormMTimesNegNormM_j * invRMS // Complete the dL/dRMS calculation

				// Calculate gradient w.r.t mean square
				// dL/d(meanSq) = dL/dRMS * dRMS/d(meanSq) = dL/dRMS * 0.5 / sqrt(meanSq+eps) = dL/dRMS * 0.5 * invRMS
				dMeanSq_j := dRMS_j * 0.5 * invRMS

				// Calculate gradient w.r.t original input m for this column
				factorInvN := 1.0 / float64(n)
				// --- Potential Chunking Start (Backward Apply m Grad) ---
				for i := 0; i < n; i += defaultChunkSize {
					end_i := i + defaultChunkSize
					if end_i > n { end_i = n }
					for row := i; row < end_i; row++ {
						// Grad from the normalization directly (dL/dm = dL/d(m_norm) * d(m_norm)/dm = dL/d(m_norm) * invRMS)
						gradMDirect := dNormM_j[row] * invRMS

						// Grad through the RMS calculation (dL/dm = dL/d(meanSq) * d(meanSq)/dm = dL/d(meanSq) * (2*m/N))
						gradMIndirect := dMeanSq_j * (2.0 * mW[row*batchSize+colOffset] * factorInvN)

						// Total gradient for m[i,j]
						totalGradM := gradMDirect + gradMIndirect
						if !math.IsNaN(totalGradM) && !math.IsInf(totalGradM, 0) {
							mDw[row*batchSize+colOffset] += totalGradM
						}
					}
				}
				// --- Potential Chunking End (Backward Apply m Grad) ---
			}

			// Apply accumulated gradients to gain parameters
			// --- Potential Chunking Start (Backward Apply gain Grad) ---
			for i := 0; i < n; i += defaultChunkSize {
				end_i := i + defaultChunkSize
				if end_i > n { end_i = n }
				for row := i; row < end_i; row++ {
					if !math.IsNaN(gainDwTemp[row]) && !math.IsInf(gainDwTemp[row], 0) {
						gainDw[row] += gainDwTemp[row]
					}
				}
			}
			// --- Potential Chunking End (Backward Apply gain Grad) ---
		}
		g.addBackward(backward)
	}
	return out
}

// --- Softmax (Chunking potentially useful for the inner hiddenSize loops) ---
func (g *Graph) Softmax(m *Mat) *Mat {
	n := m.N; batchSize := m.D
	out := NewMat(n, batchSize)
	mW := m.W; outW := out.W

	for j := 0; j < batchSize; j++ { // Iterate columns (batch)
		maxVal := -math.MaxFloat64
		colOffset := j
		// --- Potential Chunking Start (Softmax Find Max) ---
		for i := 0; i < n; i += defaultChunkSize {
			end_i := i + defaultChunkSize
			if end_i > n { end_i = n }
			maxChunk := -math.MaxFloat64
			for row := i; row < end_i; row++ {
				val := mW[row*batchSize+colOffset]
				if val > maxChunk { maxChunk = val }
			}
			if maxChunk > maxVal { maxVal = maxChunk }
		}
		// --- Potential Chunking End (Softmax Find Max) ---

		sumExp := 0.0
		expValsCol := Zeros(n) // Store exp(x - max) for stability and reuse
		// --- Potential Chunking Start (Softmax Calc Exp Sum) ---
		for i := 0; i < n; i += defaultChunkSize {
			end_i := i + defaultChunkSize
			if end_i > n { end_i = n }
			sumChunk := 0.0
			for row := i; row < end_i; row++ {
				expVal := math.Exp(mW[row*batchSize+colOffset] - maxVal)
				if math.IsNaN(expVal) || math.IsInf(expVal, 0) { expVal = 0 } // Handle potential overflow
				expValsCol[row] = expVal
				sumChunk += expVal
			}
			sumExp += sumChunk
		}
		// --- Potential Chunking End (Softmax Calc Exp Sum) ---

		invSumExp := 1.0 / (sumExp + 1e-9) // Add epsilon for numerical stability
		// Handle case where sumExp is essentially zero (e.g., large negative inputs)
		if sumExp < 1e-9 {
			invSumExp = 1.0 / float64(n) // Assign uniform probability
			// --- Potential Chunking Start (Softmax Assign Uniform) ---
			for i := 0; i < n; i += defaultChunkSize {
				end_i := i + defaultChunkSize
				if end_i > n { end_i = n }
				for row := i; row < end_i; row++ {
					outW[row*batchSize+colOffset] = invSumExp
				}
			}
			// --- Potential Chunking End (Softmax Assign Uniform) ---
		} else {
			// --- Potential Chunking Start (Softmax Assign Probs) ---
			for i := 0; i < n; i += defaultChunkSize {
				end_i := i + defaultChunkSize
				if end_i > n { end_i = n }
				for row := i; row < end_i; row++ {
					outW[row*batchSize+colOffset] = expValsCol[row] * invSumExp
				}
			}
			// --- Potential Chunking End (Softmax Assign Probs) ---
		}
	}

	if g.NeedsBackprop {
		mDw := m.Dw; outDw := out.Dw
		backward := func() {
			for j := 0; j < batchSize; j++ { // Iterate columns (batch)
				dL_dOutput_j := Zeros(n) // Gradient w.r.t. softmax output for this column
				probs_j := Zeros(n)      // Softmax probabilities for this column
				colOffset := j

				// --- Potential Chunking Start (Softmax Copy Grads/Probs) ---
				for i := 0; i < n; i += defaultChunkSize {
					end_i := i + defaultChunkSize
					if end_i > n { end_i = n }
					for row := i; row < end_i; row++ {
						dL_dOutput_j[row] = outDw[row*batchSize+colOffset]
						probs_j[row] = outW[row*batchSize+colOffset]
					}
				}
				// --- Potential Chunking End (Softmax Copy Grads/Probs) ---

				// Calculate dot product: sum(dL/dOutput_k * prob_k)
				dotProd := 0.0
				// --- Potential Chunking Start (Softmax Calc DotProd) ---
				for k := 0; k < n; k += defaultChunkSize {
					end_k := k + defaultChunkSize
					if end_k > n { end_k = n }
					dotChunk := 0.0
					for row := k; row < end_k; row++ {
						// Add safety checks
						dL_k := dL_dOutput_j[row]
						p_k := probs_j[row]
						if !math.IsNaN(dL_k) && !math.IsInf(dL_k, 0) && !math.IsNaN(p_k) && !math.IsInf(p_k, 0) {
							dotChunk += dL_k * p_k
						}
					}
					dotProd += dotChunk
				}
				// --- Potential Chunking End (Softmax Calc DotProd) ---

				if math.IsNaN(dotProd) || math.IsInf(dotProd, 0) { dotProd = 0 } // Safety check

				// Calculate gradient w.r.t. input logits: dL/dInput_i = prob_i * (dL/dOutput_i - sum(dL/dOutput_k * prob_k))
				// --- Potential Chunking Start (Softmax Apply Input Grad) ---
				for i := 0; i < n; i += defaultChunkSize {
					end_i := i + defaultChunkSize
					if end_i > n { end_i = n }
					for row := i; row < end_i; row++ {
						prob_i := probs_j[row]
						dL_dOutput_i := dL_dOutput_j[row]
						// Add safety checks
						if math.IsNaN(prob_i) || math.IsInf(prob_i, 0) || math.IsNaN(dL_dOutput_i) || math.IsInf(dL_dOutput_i, 0) {
							continue
						}
						gradInput_i := prob_i * (dL_dOutput_i - dotProd)
						if !math.IsNaN(gradInput_i) && !math.IsInf(gradInput_i, 0) {
							mDw[row*batchSize+colOffset] += gradInput_i
						}
					}
				}
				// --- Potential Chunking End (Softmax Apply Input Grad) ---
			}
		}
		g.addBackward(backward)
	}
	return out
}

// --- SoftmaxStandalone (Chunking potentially useful) ---
func SoftmaxStandalone(m *Mat) *Mat {
	// Implementation mirrors g.Softmax forward pass with chunking
	n := m.N; batchSize := m.D
	out := NewMat(n, batchSize)
	mW := m.W; outW := out.W

	for j := 0; j < batchSize; j++ {
		maxVal := -math.MaxFloat64
		colOffset := j
		// --- Chunking: Find Max ---
		for i := 0; i < n; i += defaultChunkSize {
			end_i := i + defaultChunkSize; if end_i > n { end_i = n }
			maxChunk := -math.MaxFloat64
			for row := i; row < end_i; row++ { val := mW[row*batchSize+colOffset]; if val > maxChunk { maxChunk = val } }
			if maxChunk > maxVal { maxVal = maxChunk }
		}

		s := 0.0
		expValsCol := Zeros(n)
		// --- Chunking: Calc Exp Sum ---
		for i := 0; i < n; i += defaultChunkSize {
			end_i := i + defaultChunkSize; if end_i > n { end_i = n }
			sumChunk := 0.0
			for row := i; row < end_i; row++ { expVal := math.Exp(mW[row*batchSize+colOffset] - maxVal); if math.IsNaN(expVal) || math.IsInf(expVal, 0) { expVal = 0 }; expValsCol[row] = expVal; sumChunk += expVal }
			s += sumChunk
		}

		invS := 1.0 / (s + 1e-9)
		if s < 1e-9 {
			invS = 1.0 / float64(n)
			// --- Chunking: Assign Uniform ---
			for i := 0; i < n; i += defaultChunkSize {
				end_i := i + defaultChunkSize; if end_i > n { end_i = n }
				for row := i; row < end_i; row++ { outW[row*batchSize+colOffset] = invS }
			}
		} else {
			// --- Chunking: Assign Probs ---
			for i := 0; i < n; i += defaultChunkSize {
				end_i := i + defaultChunkSize; if end_i > n { end_i = n }
				for row := i; row < end_i; row++ { outW[row*batchSize+colOffset] = expValsCol[row] * invS }
			}
		}
	}
	return out
}

// --- StackCols (Chunking less applicable - copying full columns) ---
func StackCols(g *Graph, mats []*Mat) *Mat {
	if len(mats) == 0 { log.Panic("stackCols requires a non-empty array of matrices.") }
	n := mats[0].N; numMats := len(mats); dOut := numMats
	for i := 0; i < numMats; i++ { // Assertions
		assert(mats[i] != nil, fmt.Sprintf("stackCols: Matrix %d is nil.", i))
		assert(mats[i].N == n, fmt.Sprintf("stackCols: Matrix %d has height %d, expected %d.", i, mats[i].N, n))
		assert(mats[i].D == 1, fmt.Sprintf("stackCols: Matrix %d has width %d, expected 1.", i, mats[i].D))
	}
	out := NewMat(n, dOut)
	outW := out.W
	for j := 0; j < numMats; j++ { // Iterate through matrices to stack
		matjW := mats[j].W
		destColOffset := j
		for i := 0; i < n; i++ { // Iterate through rows
			outW[i*dOut+destColOffset] = matjW[i]
		}
	}
	if g.NeedsBackprop {
		outDw := out.Dw
		backward := func() {
			for j := 0; j < numMats; j++ { // Iterate through source matrices
				matjDw := mats[j].Dw
				srcColOffset := j
				for i := 0; i < n; i++ { // Iterate through rows
					grad := outDw[i*dOut+srcColOffset]
					if !math.IsNaN(grad) && !math.IsInf(grad, 0) { // Safety check
						matjDw[i] += grad
					}
				}
			}
		}
		g.addBackward(backward)
	}
	return out
}


//======================================================================
// --- MoE MinGRU Model Definition ---
//======================================================================
// InitMoEGRU uses the actual vocab size passed to it.
func InitMoEGRU(actualVocabSize int, embeddingDim int, hiddenSizes []int, outputSize int, numExperts int) map[string]*Mat {
	// Ensure outputSize matches actualVocabSize, log warning if they differ
	if outputSize != actualVocabSize {
		log.Printf("Warning: InitMoEGRU called with outputSize %d but actualVocabSize is %d. Using actualVocabSize %d for output layer.", outputSize, actualVocabSize, actualVocabSize)
		outputSize = actualVocabSize
	}
	if actualVocabSize <= 0 {
		log.Panicf("Cannot initialize model with non-positive vocab size: %d", actualVocabSize)
	}

	log.Printf("Initializing model parameters (Actual Vocab: %d, Experts: %d)...", actualVocabSize, numExperts)
	model := make(map[string]*Mat)

	initStdDev := func(size int) float64 { return 0.08 } // Simplified He-like init

	// --- Embedding Layer ---
	log.Printf("Initializing Embedding Layer WE: %d x %d", actualVocabSize, embeddingDim)
	model["WE"] = NewRandMat(actualVocabSize, embeddingDim, 0, 0.02) // Smaller stddev for embeddings

	// --- GRU Layers ---
	layerInputSize := embeddingDim
	for d, hiddenSize := range hiddenSizes {
		log.Printf("Layer %d: Input Size %d, Hidden Size %d", d, layerInputSize, hiddenSize)
		stdGate := initStdDev(layerInputSize)
		// Gating Layer
		model[fmt.Sprintf("Wg%d", d)] = NewRandMat(numExperts, layerInputSize, 0, stdGate) // Gating weights
		model[fmt.Sprintf("bg%d", d)] = NewMat(numExperts, 1)                             // Gating bias

		// Expert Layers
		for e := 0; e < numExperts; e++ {
			stdX := initStdDev(layerInputSize)
			stdH := initStdDev(hiddenSize)
			expertSuffix := fmt.Sprintf("_exp%d", e)

			// Update gate (z)
			model[fmt.Sprintf("Wzx%d%s", d, expertSuffix)] = NewRandMat(hiddenSize, layerInputSize, 0, stdX)
			model[fmt.Sprintf("bz%d%s", d, expertSuffix)] = NewMat(hiddenSize, 1)
			// Candidate hidden state (h_candidate)
			model[fmt.Sprintf("Whx%d%s", d, expertSuffix)] = NewRandMat(hiddenSize, layerInputSize, 0, stdX) // Input to candidate
			model[fmt.Sprintf("Whh%d%s", d, expertSuffix)] = NewRandMat(hiddenSize, hiddenSize, 0, stdH)   // Hidden to candidate
			model[fmt.Sprintf("bh%d%s", d, expertSuffix)] = NewMat(hiddenSize, 1)                          // Candidate bias
		}

		// Residual connection projection (if dimensions mismatch)
		if layerInputSize != hiddenSize {
			log.Printf("  Layer %d: Adding projection %dx%d for residual connection.", d, hiddenSize, layerInputSize)
			model[fmt.Sprintf("Wp%d", d)] = NewRandMat(hiddenSize, layerInputSize, 0, initStdDev(layerInputSize))
			model[fmt.Sprintf("bp%d", d)] = NewMat(hiddenSize, 1)
		} else {
			log.Printf("  Layer %d: Residual connection dimensions match (%dx%d), no projection needed.", d, hiddenSize, layerInputSize)
		}

		// RMSNorm gain parameter for this layer's output
		log.Printf("  Layer %d: Adding RMSNorm gain parameter g_rms%d (%dx1).", d, d, hiddenSize)
		model[fmt.Sprintf("g_rms%d", d)] = NewMat(hiddenSize, 1)
		// Initialize gain to 1
		for i := range model[fmt.Sprintf("g_rms%d", d)].W { model[fmt.Sprintf("g_rms%d", d)].W[i] = 1.0 }

		layerInputSize = hiddenSize // Input size for the next layer is the output size of this one
	}

	// --- Output Layer ---
	finalHiddenSize := layerInputSize // Size of the output from the last GRU layer
	if len(hiddenSizes) > 0 { finalHiddenSize = hiddenSizes[len(hiddenSizes)-1] }
	log.Printf("Initializing Output Layer Whd: %d x %d", outputSize, finalHiddenSize)
	model["Whd"] = NewRandMat(outputSize, finalHiddenSize, 0, initStdDev(finalHiddenSize)) // Output weights
	model["bd"] = NewMat(outputSize, 1)                                                      // Output bias

	log.Println("Parameter Keys Initialized:", len(model))
	return model
}

// ForwardResult holds the outputs of the forward pass.
type ForwardResult struct {
	H [][]*Mat // Hidden states per layer, per expert [layer][expert][HiddenSize x BatchSize]
	O *Mat      // Output logits [VocabSize x BatchSize]
}

// ForwardMoEGRU computes the forward pass for one time step.
func ForwardMoEGRU(g *Graph, model map[string]*Mat, hiddenSizes []int, numExperts int, x *Mat, prevHiddenStates [][]*Mat) ForwardResult {
	currentBatchSize := x.D // Batch size can change (e.g., last batch)

	// Initialize hidden states if necessary (first time step or different batch size)
	needsInit := prevHiddenStates == nil || len(prevHiddenStates) != len(hiddenSizes)
	if !needsInit {
		for dChk := 0; dChk < len(hiddenSizes); dChk++ {
			if len(prevHiddenStates[dChk]) != numExperts { needsInit = true; break }
			// Check if any expert state in this layer needs re-init
			if len(prevHiddenStates[dChk]) > 0 { // Ensure there are experts
				if prevHiddenStates[dChk][0] == nil || prevHiddenStates[dChk][0].N != hiddenSizes[dChk] || prevHiddenStates[dChk][0].D != currentBatchSize {
					needsInit = true; break
				}
			} else { needsInit = true; break } // No experts found, needs init
		}
	}

	if needsInit {
		prevHiddenStates = make([][]*Mat, len(hiddenSizes))
		for dInit := 0; dInit < len(hiddenSizes); dInit++ {
			prevHiddenStates[dInit] = make([]*Mat, numExperts)
			for eInit := 0; eInit < numExperts; eInit++ {
				prevHiddenStates[dInit][eInit] = NewMat(hiddenSizes[dInit], currentBatchSize) // Zero-initialized
			}
		}
	}

	currentHiddenStatesLayers := make([][]*Mat, len(hiddenSizes)) // Store H for this time step
	inputToLayer := x                                             // Start with the input embeddings

	// --- Process GRU Layers ---
	for d, hiddenSize := range hiddenSizes { // Iterate through layers
		layerInputSize := inputToLayer.N
		expertOutputs := make([]*Mat, numExperts)        // Stores h_new_expert for each expert
		currentLayerExpertStates := make([]*Mat, numExperts) // Stores h_new_expert (to return)
		residualSource := inputToLayer                   // Input to this layer is used for residual connection

		// --- Gating Mechanism ---
		wgKey := fmt.Sprintf("Wg%d", d); bgKey := fmt.Sprintf("bg%d", d)
		Wg := model[wgKey]; bg := model[bgKey] // Gating weights and bias
		assert(Wg != nil && bg != nil, fmt.Sprintf("Gating weights %s or %s not found", wgKey, bgKey))
		assert(Wg.D == layerInputSize, fmt.Sprintf("Wg dim mismatch layer %d. Wg.D=%d, layerInputSize=%d", d, Wg.D, layerInputSize))

		gatingLogitsLinear := g.Mul(Wg, inputToLayer)       // Wg * x_t
		gatingLogits := g.AddBroadcastCol(gatingLogitsLinear, bg) // + bg
		gatingWeights := g.Softmax(gatingLogits)            // softmax(Wg*x_t + bg) -> [NumExperts x BatchSize]
		assert(gatingWeights.N == numExperts && gatingWeights.D == currentBatchSize, fmt.Sprintf("Gating weights dim error layer %d", d))

		// --- Expert Computations (Parallel) ---
		var wgExperts sync.WaitGroup
		wgExperts.Add(numExperts)
		for e := 0; e < numExperts; e++ {
			go func(expertIdx int) {
				defer wgExperts.Done()
				hPrevExpert := prevHiddenStates[d][expertIdx] // Hidden state from t-1 for this expert
				assert(hPrevExpert.N == hiddenSize && hPrevExpert.D == currentBatchSize, fmt.Sprintf("Prev hidden state dim error layer %d exp %d. hPrev: %dx%d, expected: %dx%d", d, expertIdx, hPrevExpert.N, hPrevExpert.D, hiddenSize, currentBatchSize))

				expertSuffix := fmt.Sprintf("_exp%d", expertIdx)
				// Get weights for this expert
				wzxKey, bzKey := fmt.Sprintf("Wzx%d%s", d, expertSuffix), fmt.Sprintf("bz%d%s", d, expertSuffix)
				whxKey, whhKey, bhKey := fmt.Sprintf("Whx%d%s", d, expertSuffix), fmt.Sprintf("Whh%d%s", d, expertSuffix), fmt.Sprintf("bh%d%s", d, expertSuffix)
				Wzx_e, bz_e := model[wzxKey], model[bzKey]
				Whx_e, Whh_e, bh_e := model[whxKey], model[whhKey], model[bhKey]
				assert(Wzx_e != nil && bz_e != nil && Whx_e != nil && Whh_e != nil && bh_e != nil, fmt.Sprintf("Missing weights L%d E%d", d, expertIdx))

				// GRU Logic for expert 'e'
				// Update gate (z_t)
				zLinear := g.Mul(Wzx_e, inputToLayer)               // Wzx * x_t
				z_t_e := g.Sigmoid(g.AddBroadcastCol(zLinear, bz_e)) // sigmoid(Wzx * x_t + bz)

				// Candidate hidden state (h_tilde_t)
				termWhx := g.Mul(Whx_e, inputToLayer)                   // Whx * x_t
				termWhh := g.Mul(Whh_e, hPrevExpert)                    // Whh * h_{t-1}
				hCandLinear := g.Add(termWhx, termWhh)                  // Whx*x_t + Whh*h_{t-1}
				hCandidate_e := g.Gelu(g.AddBroadcastCol(hCandLinear, bh_e)) // Activation(Whx*x_t + Whh*h_{t-1} + bh)

				// Combine for new hidden state (h_t)
				oneMinusZ_e := g.OneMinus(z_t_e)                               // (1 - z_t)
				term1_e := g.Eltmul(oneMinusZ_e, hPrevExpert)                  // (1 - z_t) * h_{t-1}
				term2_e := g.Eltmul(z_t_e, hCandidate_e)                       // z_t * h_tilde_t
				hNewExpert := g.Add(term1_e, term2_e)                          // h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde_t
				assert(hNewExpert.N == hiddenSize && hNewExpert.D == currentBatchSize, fmt.Sprintf("h_new_expert dim error L%d E%d", d, expertIdx))

				expertOutputs[expertIdx] = hNewExpert            // Store for weighted sum
				currentLayerExpertStates[expertIdx] = hNewExpert // Store for returning
			}(e)
		}
		wgExperts.Wait() // Wait for all expert computations to finish

		// --- Combine Expert Outputs ---
		// h_new_combined = sum(gating_weight_e * h_new_expert_e for e in experts)
		hNewCombined := g.CombineExperts(expertOutputs, gatingWeights)

		// --- Residual Connection ---
		var projectedResidual *Mat
		if layerInputSize == hiddenSize {
			// Dimensions match, use input directly
			projectedResidual = residualSource
		} else {
			// Dimensions mismatch, apply projection
			wpKey, bpKey := fmt.Sprintf("Wp%d", d), fmt.Sprintf("bp%d", d)
			Wp, bp := model[wpKey], model[bpKey]
			assert(Wp != nil && bp != nil, fmt.Sprintf("Projection Wp%d or bp%d not found.", d, d))
			projLinear := g.Mul(Wp, residualSource)
			projectedResidual = g.AddBroadcastCol(projLinear, bp)
		}
		assert(projectedResidual.N == hNewCombined.N && projectedResidual.D == hNewCombined.D, "Residual dim mismatch")
		outputWithResidual := g.Add(hNewCombined, projectedResidual) // Add residual

		// --- Layer Normalization (RMSNorm) ---
		gRMSKey := fmt.Sprintf("g_rms%d", d)
		gRMS := model[gRMSKey] // Get gain parameter for this layer
		assert(gRMS != nil && gRMS.N == hiddenSize && gRMS.D == 1, fmt.Sprintf("RMSNorm gain g_rms%d error.", d))
		normalizedOutput := g.RMSNorm(outputWithResidual, gRMS) // Apply RMSNorm
		assert(normalizedOutput.N == hiddenSize && normalizedOutput.D == currentBatchSize, fmt.Sprintf("RMSNorm output dim error L%d", d))

		// Store hidden states for this layer (needed for next time step's prevHiddenStates)
		currentHiddenStatesLayers[d] = currentLayerExpertStates
		// Output of this layer becomes input to the next
		inputToLayer = normalizedOutput
	}

	// --- Output Layer ---
	lastLayerOutput := inputToLayer // Output from the final GRU layer (after RMSNorm)
	finalHiddenSize := lastLayerOutput.N
	Whd, bd := model["Whd"], model["bd"] // Output projection weights and bias
	assert(Whd != nil && bd != nil, "Output weights Whd or bd not found")
	assert(Whd.D == finalHiddenSize, fmt.Sprintf("Output Whd dim mismatch. Whd.D=%d, finalHiddenSize=%d", Whd.D, finalHiddenSize))

	outputLogitsLinear := g.Mul(Whd, lastLayerOutput)       // Whd * h_final
	outputLogits := g.AddBroadcastCol(outputLogitsLinear, bd) // + bd
	assert(outputLogits.N == bpeActualVocabSize && outputLogits.D == currentBatchSize, fmt.Sprintf("Output logits dim error. Got %dx%d, expected %dx%d", outputLogits.N, outputLogits.D, bpeActualVocabSize, currentBatchSize))

	// Return the final logits and the hidden states computed at this step
	return ForwardResult{H: currentHiddenStatesLayers, O: outputLogits}
}


//======================================================================
// --- Model Parameter Utilities ---
//======================================================================
func GetModelParameters(model map[string]*Mat) []*Mat {
	params := make([]*Mat, 0, len(model))
	keys := make([]string, 0, len(model))
	for k := range model { keys = append(keys, k) }
	sort.Strings(keys) // Ensure consistent order for optimizer state mapping
	for _, k := range keys { params = append(params, model[k]) }
	return params
}

// ZeroModelGrads uses the updated chunked Mat.ZeroGrads method
func ZeroModelGrads(model map[string]*Mat) {
	for _, mat := range model {
		if mat != nil { // Add nil check for safety
			mat.ZeroGrads()
		}
	}
}

//======================================================================
// --- AdamW Optimizer ---
//======================================================================
type SolverAdamW struct {
	LR        float64
	Beta1     float64
	Beta2     float64
	Eps       float64
	WD        float64
	T         int
	M         map[string][]float64 // 1st moment estimate (maps parameter key to slice)
	V         map[string][]float64 // 2nd moment estimate
	paramKeys map[string]bool      // Tracks keys seen to initialize M and V
}

func NewSolverAdamW(learningRate, beta1, beta2, epsilon, weightDecay float64) *SolverAdamW {
	log.Printf("Initializing AdamW Optimizer: LR=%.e, Beta1=%.3f, Beta2=%.3f, Eps=%.e, WD=%.e",
		learningRate, beta1, beta2, epsilon, weightDecay)
	return &SolverAdamW{
		LR:        learningRate,
		Beta1:     beta1,
		Beta2:     beta2,
		Eps:       epsilon,
		WD:        weightDecay,
		T:         0,
		M:         make(map[string][]float64, 50), // Pre-allocate slightly
		V:         make(map[string][]float64, 50),
		paramKeys: make(map[string]bool, 50),
	}
}

// Step performs a parameter update using the AdamW algorithm with chunking.
func (s *SolverAdamW) Step(model map[string]*Mat) {
	s.T++
	t := float64(s.T)

	// Bias correction terms
	beta1PowT := math.Pow(s.Beta1, t)
	beta2PowT := math.Pow(s.Beta2, t)
	// Effective learning rate for this step with bias correction
	lrT := s.LR * math.Sqrt(1.0-beta2PowT) / (1.0-beta1PowT)

	// Constants for update rules
	beta1Complement := 1.0 - s.Beta1
	beta2Complement := 1.0 - s.Beta2
	effectiveWD := s.LR * s.WD // Precompute LR * WD for decay step

	// Ensure M and V buffers exist and have the correct size for all parameters
	for k, p := range model {
		if p == nil { continue } // Skip nil parameters
		paramLen := len(p.W)
		if _, exists := s.paramKeys[k]; !exists {
			s.M[k] = make([]float64, paramLen)
			s.V[k] = make([]float64, paramLen)
			s.paramKeys[k] = true
		} else if len(s.M[k]) != paramLen || len(s.V[k]) != paramLen {
			// Handle size mismatch if model structure changed unexpectedly
			log.Printf("Warning: Optimizer state size mismatch for key '%s'. Reinitializing M/V.", k)
			s.M[k] = make([]float64, paramLen)
			s.V[k] = make([]float64, paramLen)
		}
	}

	// Apply updates parameter by parameter
	for k, p := range model {
		if p == nil { continue } // Skip nil parameters

		mK := s.M[k] // Get moment buffers for this parameter
		vK := s.V[k]
		w := p.W   // Parameter weights
		dw := p.Dw // Parameter gradients

		paramLen := len(w)

		// Process in chunks
		for i := 0; i < paramLen; i += defaultChunkSize {
			end := i + defaultChunkSize
			if end > paramLen {
				end = paramLen
			}

			// Inner loop over the chunk
			for j := i; j < end; j++ {
				grad := dw[j]

				// Handle NaN/Inf gradients - skip update for this element
				if math.IsNaN(grad) || math.IsInf(grad, 0) {
					dw[j] = 0.0 // Clear the bad gradient
					// Optionally reset moments if they became NaN/Inf, though AdaMW tends to be robust
					if math.IsNaN(mK[j]) || math.IsInf(mK[j], 0) { mK[j] = 0 }
					if math.IsNaN(vK[j]) || math.IsInf(vK[j], 0) { vK[j] = 0 }
					continue
				}

				// Update biased first moment estimate
				mK[j] = s.Beta1*mK[j] + beta1Complement*grad
				// Update biased second raw moment estimate
				vK[j] = s.Beta2*vK[j] + beta2Complement*(grad*grad)

				// Compute the update term (Adam part)
				// Denominator: sqrt(v_hat) + epsilon
				denom := math.Sqrt(vK[j]) + s.Eps
				update := lrT * mK[j] / denom

				// Apply AdamW update: param = param - update - effectiveWD * param
				// Note: Weight decay is applied *after* the momentum update, typical for AdamW
				w[j] -= update + (effectiveWD * w[j])
			}
		}
		// Gradients are zeroed outside the loop after processing all parameters
		// p.ZeroGrads() // Use ZeroModelGrads after the loop
	}
	// Zero all gradients after the update step
	ZeroModelGrads(model)
}

// StepParallel performs parameter updates in parallel for large models
// (Implementation uses Step internally for now, could be enhanced)
func (s *SolverAdamW) StepParallel(model map[string]*Mat, numWorkers int) {
    // Basic parallelization: distribute parameter updates across workers
    // More advanced: parallelize *within* large parameter updates (like OptimizedStep)

    s.T++
    t := float64(s.T)
    beta1PowT := math.Pow(s.Beta1, t)
    beta2PowT := math.Pow(s.Beta2, t)
    lrT := s.LR * math.Sqrt(1.0-beta2PowT) / (1.0-beta1PowT)
    beta1Complement := 1.0 - s.Beta1
    beta2Complement := 1.0 - s.Beta2
    effectiveWD := s.LR * s.WD

    if numWorkers <= 0 {
        numWorkers = runtime.NumCPU()
    }
    if numWorkers > len(model) { // Don't need more workers than parameters
        numWorkers = len(model)
    }
    if numWorkers <= 0 { numWorkers = 1} // Ensure at least one worker

    // Initialize missing buffers (sequentially first)
    for k, p := range model {
        if p == nil { continue }
        paramLen := len(p.W)
        if _, exists := s.paramKeys[k]; !exists {
            s.M[k] = make([]float64, paramLen)
            s.V[k] = make([]float64, paramLen)
            s.paramKeys[k] = true
        } else if len(s.M[k]) != paramLen || len(s.V[k]) != paramLen {
            s.M[k] = make([]float64, paramLen)
            s.V[k] = make([]float64, paramLen)
        }
    }

    // Create a list of parameter keys to distribute
    keys := make([]string, 0, len(model))
    for k := range model {
        if model[k] != nil { // Only include non-nil parameters
             keys = append(keys, k)
        }
    }

    var wg sync.WaitGroup
    workChan := make(chan string, len(keys)) // Buffered channel

    // Start workers
    for i := 0; i < numWorkers; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            for k := range workChan { // Process keys from the channel
                p := model[k]
                mK := s.M[k]
                vK := s.V[k]
                w := p.W
                dw := p.Dw
                paramLen := len(w)

                // Process in chunks (same as sequential Step)
                for i := 0; i < paramLen; i += defaultChunkSize {
                    end := i + defaultChunkSize
                    if end > paramLen { end = paramLen }
                    for j := i; j < end; j++ {
                        grad := dw[j]
                        if math.IsNaN(grad) || math.IsInf(grad, 0) {
                            dw[j] = 0.0
							if math.IsNaN(mK[j]) || math.IsInf(mK[j], 0) { mK[j] = 0 }
							if math.IsNaN(vK[j]) || math.IsInf(vK[j], 0) { vK[j] = 0 }
                            continue
                        }
                        mK[j] = s.Beta1*mK[j] + beta1Complement*grad
                        vK[j] = s.Beta2*vK[j] + beta2Complement*(grad*grad)
                        denom := math.Sqrt(vK[j]) + s.Eps
                        update := lrT * mK[j] / denom
                        w[j] -= update + (effectiveWD * w[j])
                    }
                }
                // Gradients are zeroed globally after all workers finish
            }
        }()
    }

    // Feed keys to workers
    for _, k := range keys {
        workChan <- k
    }
    close(workChan) // Signal no more work

    wg.Wait() // Wait for all workers to finish

    // Zero all gradients after the parallel update step
    ZeroModelGrads(model)
}


// OptimizedStep decides whether to use parallel or sequential update.
func (s *SolverAdamW) OptimizedStep(model map[string]*Mat) {
	// Threshold for deciding to use parallel execution
	// This depends heavily on the machine and model size.
	// Consider total parameters or size of largest parameter.
	const parallelThresholdParams = 500_000 // Example threshold: 500k total parameters
	const largeMatrixThreshold = 100_000    // Example: Matrix with > 100k elements

	useParallel := false
	totalParams := 0
	maxParamSize := 0

	for _, p := range model {
		if p == nil { continue }
		size := len(p.W)
		totalParams += size
		if size > maxParamSize {
			maxParamSize = size
		}
	}

	// Decide based on thresholds
	if totalParams > parallelThresholdParams || maxParamSize > largeMatrixThreshold {
		useParallel = true
	}

	if useParallel {
		// log.Println("Using parallel optimizer step") // Optional: for debugging
		s.StepParallel(model, 0) // Use default number of workers (NumCPU)
	} else {
		// log.Println("Using sequential optimizer step") // Optional: for debugging
		s.Step(model)
	}
}


// GetState returns the current state for serialization
func (s *SolverAdamW) GetState() SerializableSolverState {
	// Create copies of the moment maps to avoid concurrent modification issues
	// if the state is used elsewhere while the solver continues.
	mCopy := make(map[string][]float64, len(s.M))
	for k, v := range s.M {
		vCopy := make([]float64, len(v))
		copy(vCopy, v)
		mCopy[k] = vCopy
	}
	vCopyMap := make(map[string][]float64, len(s.V))
	for k, v := range s.V {
		vCopy := make([]float64, len(v))
		copy(vCopy, v)
		vCopyMap[k] = vCopy
	}

	return SerializableSolverState{
		LR:    s.LR,
		Beta1: s.Beta1,
		Beta2: s.Beta2,
		Eps:   s.Eps,
		WD:    s.WD,
		T:     s.T,
		M:     mCopy,
		V:     vCopyMap,
	}
}

// LoadState loads optimizer state from serialized data
func (s *SolverAdamW) LoadState(state SerializableSolverState) {
	s.LR = state.LR
	s.Beta1 = state.Beta1
	s.Beta2 = state.Beta2
	s.Eps = state.Eps
	s.WD = state.WD
	s.T = state.T
	// Take ownership of the loaded maps (assuming they are not used elsewhere)
	s.M = state.M
	s.V = state.V

	// Rebuild the parameter keys map for consistency checks during Step
	keyCount := len(s.M)
	s.paramKeys = make(map[string]bool, keyCount)
	for k := range s.M {
		s.paramKeys[k] = true
	}

	log.Printf("Optimizer state loaded. T=%d, LR=%.e, Beta1=%.3f, Beta2=%.3f, Eps=%.e, WD=%.e, Keys=%d",
		s.T, s.LR, s.Beta1, s.Beta2, s.Eps, s.WD, keyCount)
}
//======================================================================
// --- Helper: Create One-Hot Batch Matrix --- (Keep as is or remove if unused)
//======================================================================
func createOneHotBatch(tokenIDs []int, vocabSize int) *Mat {
	currentBatchSize := len(tokenIDs)
	assert(currentBatchSize > 0, "createOneHotBatch requires at least one token ID")
	batchVec := NewMat(vocabSize, currentBatchSize) // VocabSize x BatchSize
	for j, tokenID := range tokenIDs { // Iterate through batch
		if tokenID >= 0 && tokenID < vocabSize {
			batchVec.Set(tokenID, j, 1.0) // Set the 'hot' entry for this batch item
		} else if tokenID != -1 { // Allow -1 as a skip/padding indicator
			log.Printf("Warning: Index %d out of bounds for one-hot vector size %d in batch item %d.", tokenID, vocabSize, j)
		}
	}
	return batchVec
}

//======================================================================
// --- BPE Training Functionality (Now called only in bpe-train mode) ---
//======================================================================
// handleBPETraining orchestrates BPE training and saving.
func handleBPETraining() error {
	if flagBPEData == "" {
		return errors.New("BPE training mode requires --bpe-data flag (path to corpus)")
	}
	if flagBPEOutputPath == "" {
		return errors.New("BPE training mode requires --bpe-output flag (path to save trained BPE state)")
	}
	if flagBPEVocabSize <= len(BpeSpecialTokens) {
		return fmt.Errorf("BPE vocab size (%d) must be greater than the number of special tokens (%d)", flagBPEVocabSize, len(BpeSpecialTokens))
	}

	log.Printf("Status: Running in BPE Training mode...")
	log.Printf("  Corpus: %s", flagBPEData)
	log.Printf("  Output: %s", flagBPEOutputPath)
	log.Printf("  Target Vocab Size: %d", flagBPEVocabSize)

	bpeInstance := NewBPE(BpeSpecialTokens) // Create a new BPE instance

	log.Println("\n--- Training BPE Tokenizer ---")
	dataBytes, err := ioutil.ReadFile(flagBPEData)
	if err != nil {
		return fmt.Errorf("failed to read BPE data file '%s': %w", flagBPEData, err)
	}
	bpeCorpus := string(dataBytes)
	if len(strings.TrimSpace(bpeCorpus)) == 0 {
		return fmt.Errorf("BPE data file '%s' is empty or contains only whitespace", flagBPEData)
	}
	log.Printf("Successfully loaded %d bytes of BPE training data from %s", len(bpeCorpus), flagBPEData)

	bpeLogWrapper := func(msg string) { log.Println("BPE:", msg) }
	bpeInstance.Train(bpeCorpus, flagBPEVocabSize, false, bpeLogWrapper) // Train the instance

	actualVocabSize := len(bpeInstance.vocabArray)
	if actualVocabSize == 0 {
		return errors.New("BPE training resulted in zero vocab size")
	}
	log.Printf("BPE training complete. Actual Vocab Size: %d", actualVocabSize)

	// Save the trained BPE state
	err = SaveBPEState(bpeInstance, flagBPEOutputPath)
	if err != nil {
		return fmt.Errorf("failed to save trained BPE state to '%s': %w", flagBPEOutputPath, err)
	}

	log.Println("Status: BPE training and saving finished successfully.")
	return nil // Indicate success
}

//======================================================================
// --- Model Data Preparation (Batching and Shuffling) ---
//======================================================================
// prepareModelData now takes the BPE instance as an argument
func prepareModelData(modelDataPath string, bpeInstance *BPE) (bool, error) {
	log.Printf("Status: Preparing model training data from file '%s'...", modelDataPath)
	log.Println("\n--- Preparing Model Data ---")
	batches = [][]TrainingSample{} // Clear previous batches

	if bpeInstance == nil || len(bpeInstance.vocabArray) == 0 {
		return false, errors.New("BPE tokenizer is not initialized or loaded before preparing model data")
	}
	currentBPEVocabSize := len(bpeInstance.vocabArray) // Use size from the instance
	log.Printf("Using provided BPE tokenizer with %d vocab size.", currentBPEVocabSize)

	dataBytes, err := ioutil.ReadFile(modelDataPath)
	if err != nil {
		log.Printf("Status: Error: Failed to read model data file '%s'", modelDataPath)
		return false, fmt.Errorf("failed to read model data file '%s': %w", modelDataPath, err)
	}
	modelText := string(dataBytes)
	if len(strings.TrimSpace(modelText)) == 0 {
		return false, fmt.Errorf("model data file '%s' is empty or contains only whitespace", modelDataPath)
	}
	log.Printf("Successfully loaded %d bytes of model training data from %s", len(modelText), modelDataPath)

	encodedTextIDs := bpeInstance.Encode(modelText) // Use the passed BPE instance
	log.Printf("Encoded model text -> %d tokens.", len(encodedTextIDs))

	if len(encodedTextIDs) <= seqLength {
		err := fmt.Errorf("error: Encoded model text length (%d) is not greater than sequence length (%d). Cannot create training samples.", len(encodedTextIDs), seqLength)
		log.Println("Status: Error:", err)
		return false, err
	}

	allSamples := []TrainingSample{}
	for i := 0; i <= len(encodedTextIDs)-seqLength-1; i++ {
		inputSeqIDs := encodedTextIDs[i : i+seqLength]
		targetSeqIDs := encodedTextIDs[i+1 : i+seqLength+1]
		// Basic validation: ensure sequences have expected length
		if len(inputSeqIDs) != seqLength || len(targetSeqIDs) != seqLength {
			log.Printf("Warning: Skipping sample at index %d due to unexpected sequence length (input: %d, target: %d, expected: %d)", i, len(inputSeqIDs), len(targetSeqIDs), seqLength)
			continue
		}
		allSamples = append(allSamples, TrainingSample{
			Input:  append([]int{}, inputSeqIDs...), // Deep copy slices
			Target: append([]int{}, targetSeqIDs...), // Deep copy slices
		})
	}


	log.Println("Total individual sequences generated:", len(allSamples))
	if len(allSamples) == 0 {
		return false, errors.New("no training sequences generated despite sufficient text length")
	}

	currentBatchSize := batchSize
	if len(allSamples) < currentBatchSize {
		log.Printf("Warning: Number of samples (%d) is less than configured batch size (%d). Adjusting batch size for this run.", len(allSamples), currentBatchSize)
		currentBatchSize = len(allSamples)
		if currentBatchSize == 0 {
			return false, errors.New("adjusted batch size became zero (no samples)")
		}
	}

	// Shuffle samples before batching
	rand.Shuffle(len(allSamples), func(i, j int) {
		allSamples[i], allSamples[j] = allSamples[j], allSamples[i]
	})
	log.Println("Shuffled training samples.")

	// Create batches
	numBatches := len(allSamples) / currentBatchSize
	batches = make([][]TrainingSample, 0, numBatches) // Pre-allocate capacity
	for i := 0; i < numBatches; i++ {
		start := i * currentBatchSize
		end := start + currentBatchSize
		// Note: Slicing creates a view; we need to copy if the underlying array is modified
		// However, since we shuffle *before* batching, a view is okay here.
		batch := allSamples[start:end]
		if len(batch) > 0 { // Ensure batch is not empty
			batches = append(batches, batch)
		}
	}


	leftoverCount := len(allSamples) % currentBatchSize
	if leftoverCount > 0 {
		log.Printf("Info: Discarding %d leftover samples that don't form a full batch.", leftoverCount)
		// Optionally, handle partial batches:
		// lastBatch := allSamples[numBatches*currentBatchSize:]
		// if len(lastBatch) > 0 {
		//     batches = append(batches, lastBatch)
		//     log.Printf("Created %d batches (last one partial size %d).", len(batches), len(lastBatch))
		// } else {
		//     log.Printf("Created %d full batches.", len(batches))
		// }
	} else {
		log.Printf("Created %d full batches of size %d.", len(batches), currentBatchSize)
	}


	if len(batches) == 0 {
		return false, errors.New("no batches created")
	}

	log.Println("Status: Model data preparation complete.")
	return true, nil
}

//======================================================================
// --- Validation Data Preparation ---
//======================================================================
// prepareValidationData now takes the BPE instance as an argument
func prepareValidationData(validationDataPath string, bpeInstance *BPE) (bool, error) {
	log.Printf("Status: Preparing validation data from file '%s'...", validationDataPath)
	log.Println("\n--- Preparing Validation Data ---")
	validationBatches = [][]TrainingSample{} // Clear previous validation batches

	if bpeInstance == nil || len(bpeInstance.vocabArray) == 0 {
		return false, errors.New("BPE tokenizer is not initialized or loaded before preparing validation data")
	}
	currentBPEVocabSize := len(bpeInstance.vocabArray)
	log.Printf("Using provided BPE tokenizer with %d vocab size for validation.", currentBPEVocabSize)

	dataBytes, err := ioutil.ReadFile(validationDataPath)
	if err != nil {
		if os.IsNotExist(err) {
			log.Printf("Info: Validation data file '%s' not found. Skipping validation.", validationDataPath)
			return true, nil // Not an error if file just doesn't exist
		}
		log.Printf("Status: Error: Failed to read validation data file '%s'", validationDataPath)
		return false, fmt.Errorf("failed to read validation data file '%s': %w", validationDataPath, err)
	}
	validationText := string(dataBytes)
	if len(strings.TrimSpace(validationText)) == 0 {
		log.Printf("Warning: Validation data file '%s' is empty or contains only whitespace. No validation data prepared.", validationDataPath)
		return true, nil // Not an error, just no data
	}
	log.Printf("Successfully loaded %d bytes of validation data from %s", len(validationText), validationDataPath)

	encodedTextIDs := bpeInstance.Encode(validationText) // Use the passed BPE instance
	log.Printf("Encoded validation text -> %d tokens.", len(encodedTextIDs))

	if len(encodedTextIDs) <= seqLength {
		log.Printf("Warning: Encoded validation text length (%d) is not greater than sequence length (%d). Cannot create validation samples.", len(encodedTextIDs), seqLength)
		return true, nil // Not a fatal error
	}

	allValidationSamples := []TrainingSample{}
	for i := 0; i <= len(encodedTextIDs)-seqLength-1; i++ {
		inputSeqIDs := encodedTextIDs[i : i+seqLength]
		targetSeqIDs := encodedTextIDs[i+1 : i+seqLength+1]
		if len(inputSeqIDs) != seqLength || len(targetSeqIDs) != seqLength {
			log.Printf("Warning: Skipping validation sample at index %d due to unexpected sequence length.", i)
			continue
		}
		allValidationSamples = append(allValidationSamples, TrainingSample{
			Input:  append([]int{}, inputSeqIDs...), // Deep copy
			Target: append([]int{}, targetSeqIDs...), // Deep copy
		})
	}

	log.Println("Total individual validation sequences generated:", len(allValidationSamples))
	if len(allValidationSamples) == 0 {
		log.Println("Warning: No validation sequences generated.")
		return true, nil // Not fatal
	}

	currentBatchSize := batchSize
	if len(allValidationSamples) < currentBatchSize {
		log.Printf("Warning: Number of validation samples (%d) is less than configured batch size (%d). Using a smaller batch size for validation.", len(allValidationSamples), currentBatchSize)
		currentBatchSize = len(allValidationSamples)
		if currentBatchSize == 0 {
			return true, nil // No samples, no batches
		}
	}

	// Create validation batches (no shuffling needed)
	numBatches := len(allValidationSamples) / currentBatchSize
	validationBatches = make([][]TrainingSample, 0, numBatches)
	for i := 0; i < numBatches; i++ {
		start := i * currentBatchSize
		end := start + currentBatchSize
		batch := allValidationSamples[start:end]
		if len(batch) > 0 {
			validationBatches = append(validationBatches, batch)
		}
	}

	leftoverCount := len(allValidationSamples) % currentBatchSize
	if leftoverCount > 0 {
		log.Printf("Info: Discarding %d leftover validation samples that don't form a full batch.", leftoverCount)
		// Optionally handle partial validation batches
		// lastBatch := allValidationSamples[numBatches*currentBatchSize:]
		// if len(lastBatch) > 0 { validationBatches = append(validationBatches, lastBatch) }
	}

	if len(validationBatches) > 0 {
		log.Printf("Created %d validation batches of size up to %d.", len(validationBatches), currentBatchSize)
	} else {
		log.Println("Warning: No validation batches created.")
	}


	log.Println("Status: Validation data preparation complete.")
	return true, nil
}


//======================================================================
// --- Checkpointing Structures and Functions (MODEL ONLY) ---
//======================================================================
// SerializableMat remains the same
type SerializableMat struct { N int; D int; W []float64; Dw []float64 }

// SerializableSolverState remains the same
type SerializableSolverState struct { LR float64; Beta1 float64; Beta2 float64; Eps float64; WD float64; T int; M map[string][]float64; V map[string][]float64 }

// Checkpoint struct NO LONGER CONTAINS BPE state or config.
type Checkpoint struct {
	Epoch          int
	ModelParams    map[string]SerializableMat
	OptimizerState SerializableSolverState
	Config         struct { // Stores model architecture and training hyperparams *used* for this checkpoint
		EmbeddingDimension int
		NumExperts         int
		TrainSeqLength     int
		BatchSize          int
		Epochs             int // Total epochs configured for the run that saved this
		MaxResponseLength  int
		LearningRate       float64 // LR at the time of saving
		WeightDecay        float64 // WD at the time of saving
		EpsilonRMSNorm     float64
		EpsilonAdamW       float64 // Epsilon for AdamW at time of saving
		GradientClipValue  float64
		HiddenSizes        []int   // Explicitly store the hidden sizes array
	}
}

// matToSerializable potentially use chunking for copy if needed
func matToSerializable(m *Mat) SerializableMat {
	wCopy := make([]float64, len(m.W))
	dwCopy := make([]float64, len(m.Dw))
	copy(wCopy, m.W)   // TODO: Consider chunking for very large W
	copy(dwCopy, m.Dw) // TODO: Consider chunking for very large Dw
	return SerializableMat{ N: m.N, D: m.D, W: wCopy, Dw: dwCopy, }
}

// serializableToMat potentially use chunking for copy if needed
func serializableToMat(sm SerializableMat) *Mat {
	m := NewMat(sm.N, sm.D)
	copy(m.W, sm.W) // TODO: Consider chunking for very large W
	if len(sm.Dw) == len(m.Dw) {
		copy(m.Dw, sm.Dw) // TODO: Consider chunking for very large Dw
	} else if len(sm.Dw) != 0 {
		log.Printf("Warning: Checkpoint Dw size (%d) mismatch for matrix %dx%d (expected %d), gradients not loaded.", len(sm.Dw), sm.N, sm.D, len(m.Dw))
	}
	return m
}

// --- Gob Type Registration ---
func init() {
	gob.Register(Checkpoint{})
	gob.Register(SerializableMat{})
	gob.Register(SerializableSolverState{})
	gob.Register(map[string]SerializableMat{})
	gob.Register(map[string][]float64{})
	// BPE related types need registration for BPE state saving/loading
	gob.Register(BPESavedState{})
	gob.Register(map[string]MergeInfo{})
	gob.Register(MergeInfo{})
}

// saveCheckpoint saves the MODEL and OPTIMIZER state using gob.
func saveCheckpoint(epoch int, model map[string]*Mat, solver *SolverAdamW, path string) error {
	log.Printf("Saving model checkpoint for epoch %d to %s...", epoch, path)

	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create checkpoint directory %s: %w", dir, err)
	}

	serializableModel := make(map[string]SerializableMat)
	for k, v := range model {
		if v != nil { // Add nil check
			serializableModel[k] = matToSerializable(v)
		}
	}


	// Get optimizer state (creates copies internally)
	optimizerState := solver.GetState()

	// Create Checkpoint struct (without BPE)
	checkpoint := Checkpoint{
		Epoch:          epoch,
		ModelParams:    serializableModel,
		OptimizerState: optimizerState,
	}
	// Add config values *from the current global state*
	checkpoint.Config.EmbeddingDimension = embeddingDimension // Use global var
	checkpoint.Config.NumExperts = numExperts             // Use global var
	checkpoint.Config.TrainSeqLength = seqLength             // Use global var
	checkpoint.Config.BatchSize = batchSize             // Use global var
	checkpoint.Config.Epochs = flagEpochs             // Total epochs for this run
	checkpoint.Config.MaxResponseLength = flagMaxResponseLength
	checkpoint.Config.LearningRate = solver.LR        // Current LR from solver
	checkpoint.Config.WeightDecay = solver.WD         // Current WD from solver
	checkpoint.Config.EpsilonRMSNorm = flagEpsilonRMSNorm // Use global var/flag
	checkpoint.Config.EpsilonAdamW = solver.Eps         // Current Eps from solver
	checkpoint.Config.GradientClipValue = flagGradientClipValue // Use global var/flag
	checkpoint.Config.HiddenSizes = append([]int{}, hiddenSizes...) // Use global var (copy)

	// Write to file atomically using gob
	tempPath := path + ".tmp"
	file, err := os.Create(tempPath)
	if err != nil { return fmt.Errorf("failed to create temporary checkpoint file %s: %w", tempPath, err) }
	// Ensure file is closed even if encoding fails
	defer func() {
        file.Close()
        if err != nil { // Remove temp file on error
            _ = os.Remove(tempPath)
        }
    }()


	encoder := gob.NewEncoder(file)
	err = encoder.Encode(checkpoint)
	if err != nil { return fmt.Errorf("failed to encode checkpoint data to %s: %w", tempPath, err) }


	// Close explicitly before rename (defer handles error case)
	if err = file.Close(); err != nil {	return fmt.Errorf("failed to close temporary checkpoint file %s before rename: %w", tempPath, err) }


	err = os.Rename(tempPath, path)
	if err != nil { return fmt.Errorf("failed to rename temporary checkpoint file to %s: %w", path, err) }

	log.Printf("Model checkpoint saved successfully to %s", path)
	return nil
}

// loadCheckpoint loads MODEL and OPTIMIZER state using gob and updates relevant global config vars.
// It now takes the expected vocab size (from the separately loaded BPE) for validation.
func loadCheckpoint(path string, expectedVocabSize int) (startEpoch int, loadedModel map[string]*Mat, loadedSolver *SolverAdamW, err error) {
	log.Printf("Loading model checkpoint from %s...", path)

	file, err := os.Open(path)
	if err != nil { err = fmt.Errorf("failed to open checkpoint file %s: %w", path, err); return }
	defer file.Close()

	var checkpoint Checkpoint
	decoder := gob.NewDecoder(file)
	err = decoder.Decode(&checkpoint)
	if err != nil { err = fmt.Errorf("failed to decode gob checkpoint data from %s: %w", path, err); return }

	// Validate Config against current *flag* values (informational)
	log.Println("Validating checkpoint configuration against current flag settings...")
	configMismatch := false
	// Compare loaded config to *flags* where applicable, use loaded values for globals later
	if checkpoint.Config.EmbeddingDimension != flagEmbeddingDimension { log.Printf("Warning: Checkpoint EmbeddingDimension (%d) differs from current flag (%d)", checkpoint.Config.EmbeddingDimension, flagEmbeddingDimension); configMismatch = true }
	if len(checkpoint.Config.HiddenSizes) != flagGRULayers { log.Printf("Warning: Checkpoint GRULayers (%d) differs from current flag (%d)", len(checkpoint.Config.HiddenSizes), flagGRULayers); configMismatch = true }
	// Check hidden size only if layers exist
	if len(checkpoint.Config.HiddenSizes) > 0 && checkpoint.Config.HiddenSizes[0] != flagGRUHiddenSize { log.Printf("Warning: Checkpoint GRUHiddenSize (%d) differs from current flag (%d)", checkpoint.Config.HiddenSizes[0], flagGRUHiddenSize); configMismatch = true } else if len(checkpoint.Config.HiddenSizes) == 0 && flagGRULayers > 0 { log.Printf("Warning: Checkpoint has 0 GRU layers but flag specifies %d layers.", flagGRULayers); configMismatch = true}
	if checkpoint.Config.NumExperts != flagNumExperts { log.Printf("Warning: Checkpoint NumExperts (%d) differs from current flag (%d)", checkpoint.Config.NumExperts, flagNumExperts); configMismatch = true }
	if checkpoint.Config.TrainSeqLength != flagTrainSeqLength { log.Printf("Warning: Checkpoint TrainSeqLength (%d) differs from current flag (%d)", checkpoint.Config.TrainSeqLength, flagTrainSeqLength); configMismatch = true }
	// Compare floats with tolerance
	if math.Abs(checkpoint.Config.LearningRate-flagLearningRate) > 1e-9 { log.Printf("Info: Checkpoint LearningRate (%.e) differs from current flag (%.e). Optimizer state will override.", checkpoint.Config.LearningRate, flagLearningRate) } // Informational
	if math.Abs(checkpoint.Config.WeightDecay-flagWeightDecay) > 1e-9 { log.Printf("Info: Checkpoint WeightDecay (%.e) differs from current flag (%.e). Optimizer state will override.", checkpoint.Config.WeightDecay, flagWeightDecay) } // Informational
    if math.Abs(checkpoint.Config.EpsilonRMSNorm-flagEpsilonRMSNorm) > 1e-9 { log.Printf("Warning: Checkpoint EpsilonRMSNorm (%.e) differs from current flag (%.e).", checkpoint.Config.EpsilonRMSNorm, flagEpsilonRMSNorm); configMismatch = true }
	if math.Abs(checkpoint.Config.EpsilonAdamW-flagEpsilonAdamW) > 1e-9 { log.Printf("Info: Checkpoint EpsilonAdamW (%.e) differs from current flag (%.e). Optimizer state will override.", checkpoint.Config.EpsilonAdamW, flagEpsilonAdamW) } // Informational
    if math.Abs(checkpoint.Config.GradientClipValue-flagGradientClipValue) > 1e-9 { log.Printf("Warning: Checkpoint GradientClipValue (%.2f) differs from current flag (%.2f).", checkpoint.Config.GradientClipValue, flagGradientClipValue); configMismatch = true }


	if configMismatch { log.Println("Configuration mismatch detected. Checkpoint values will be used for model structure and optimizer state. Current flags may affect subsequent training if applicable (e.g., total epochs).") } else { log.Println("Checkpoint configuration broadly matches current flag settings.") }

	// Reconstruct Model
	loadedModel = make(map[string]*Mat)
	for k, sm := range checkpoint.ModelParams { loadedModel[k] = serializableToMat(sm) }
	log.Printf("Loaded %d model parameters.", len(loadedModel))

	// *** VOCAB SIZE VALIDATION ***
	// Check if the loaded model dimensions match the expected vocab size from the BPE file
	if we, ok := loadedModel["WE"]; ok {
		if we.N != expectedVocabSize {
			err = fmt.Errorf("FATAL: Vocab size mismatch! Checkpoint embedding layer (WE) has %d rows (vocab size), but the loaded BPE file has %d tokens. Ensure the correct BPE file (--bpe-path) is used with this checkpoint.", we.N, expectedVocabSize)
			return // Return immediately with the fatal error
		}
	} else {
		err = errors.New("FATAL: Checkpoint is missing the embedding layer (WE). Cannot validate vocab size.")
		return
	}
	if whd, ok := loadedModel["Whd"]; ok {
		if whd.N != expectedVocabSize {
			// Log a warning but consider if this should be fatal.
			// It *might* be okay if only the embedding was used for transfer learning,
			// but usually output should match. Let's make it fatal for safety.
			err = fmt.Errorf("FATAL: Vocab size mismatch! Checkpoint output layer (Whd) has %d rows, but the loaded BPE file has %d tokens.", whd.N, expectedVocabSize)
			return
		}
	} else {
		err = errors.New("FATAL: Checkpoint is missing the output layer head (Whd). Cannot validate vocab size.")
		return
	}
	log.Printf("Checkpoint vocab size (%d) matches loaded BPE vocab size.", expectedVocabSize)

	// Reconstruct Optimizer
	loadedSolver = NewSolverAdamW(
		checkpoint.OptimizerState.LR,    // Use LR from checkpoint state
		checkpoint.OptimizerState.Beta1,
		checkpoint.OptimizerState.Beta2,
		checkpoint.OptimizerState.Eps,   // Use Eps from checkpoint state
		checkpoint.OptimizerState.WD,    // Use WD from checkpoint state
	)
	loadedSolver.LoadState(checkpoint.OptimizerState) // Load M, V, and T

	// Update Global Configuration Variables from Checkpoint Config (critical for resuming)
	log.Println("Applying checkpoint configuration to runtime variables...")
	embeddingDimension = checkpoint.Config.EmbeddingDimension
	hiddenSizes = append([]int{}, checkpoint.Config.HiddenSizes...) // Deep copy
	numExperts = checkpoint.Config.NumExperts
	seqLength = checkpoint.Config.TrainSeqLength
	batchSize = checkpoint.Config.BatchSize
	flagMaxResponseLength = checkpoint.Config.MaxResponseLength // Update flag value itself
	flagEpsilonRMSNorm = checkpoint.Config.EpsilonRMSNorm   // Update flag value
	flagGradientClipValue = checkpoint.Config.GradientClipValue // Update flag value
	// Note: LR, WD, EpsAdamW are implicitly set by loading the solver state above.
	// Note: flagEpochs is NOT updated here; the flag determines the *target* epoch count for the current run.

	startEpoch = checkpoint.Epoch + 1 // Training resumes from the *next* epoch
	log.Printf("Model checkpoint loaded successfully. Configuration updated. Resuming from epoch %d.", startEpoch)
	return // Returns named return values (startEpoch, loadedModel, loadedSolver, err=nil)
}

//======================================================================
// --- Validation Loss Calculation ---
//======================================================================
// calculateValidationLoss uses global bpeActualVocabSize
func calculateValidationLoss(model map[string]*Mat, valBatches [][]TrainingSample) (float64, error) {
	if len(valBatches) == 0 {
		log.Println("Info: No validation batches available to calculate loss.")
		return 0.0, nil
	}
	if model == nil { return 0.0, errors.New("validation loss calculation called but model is nil") }
	if bpeActualVocabSize <= 0 { return 0.0, errors.New("validation loss calculation called but BPE vocab size is zero") }

	log.Printf("Status: Calculating validation loss on %d batches...", len(valBatches))
	startTime := time.Now()
	totalValidationLoss := 0.0
	totalValidValidationSteps := 0

	for batchIndex, batch := range valBatches {
		currentBatchSize := len(batch)
		if currentBatchSize == 0 { continue }

		g := NewGraph(false) // No backprop needed for validation
		var hiddenStates [][]*Mat = nil // Reset hidden state for each batch

		batchLoss := 0.0
		validStepsInBatch := 0

		// Iterate through the sequence length for this batch
		for t := 0; t < seqLength; t++ {
			inputTokenIDs := make([]int, currentBatchSize)
			targetTokenIDs := make([]int, currentBatchSize)
			hasValidTargetInStep := false

			// Prepare input and target IDs for this time step
			for i := 0; i < currentBatchSize; i++ {
				if t < len(batch[i].Input) && t < len(batch[i].Target) {
					// Input ID - Check bounds
					if batch[i].Input[t] >= 0 && batch[i].Input[t] < bpeActualVocabSize {
						inputTokenIDs[i] = batch[i].Input[t]
					} else {
						inputTokenIDs[i] = -1 // Mark as invalid/padding
						// log.Printf("Debug: Invalid input token ID %d at batch %d, item %d, step %d", batch[i].Input[t], batchIndex, i, t)
					}
					// Target ID - Check bounds
					if batch[i].Target[t] >= 0 && batch[i].Target[t] < bpeActualVocabSize {
						targetTokenIDs[i] = batch[i].Target[t]
						hasValidTargetInStep = true // We have at least one valid target to calculate loss for
					} else {
						targetTokenIDs[i] = -1 // Mark as invalid/padding
						// log.Printf("Debug: Invalid target token ID %d at batch %d, item %d, step %d", batch[i].Target[t], batchIndex, i, t)
					}
				} else {
					// Sequence shorter than seqLength for this sample
					inputTokenIDs[i] = -1
					targetTokenIDs[i] = -1
				}
			}

			// If no valid targets in this step across the batch, skip forward pass
			if !hasValidTargetInStep { continue }

			// Perform forward pass for this time step
			xBatch := g.Lookup(model["WE"], inputTokenIDs) // Get embeddings [EmbedDim x BatchSize]
			forwardResult := ForwardMoEGRU(g, model, hiddenSizes, numExperts, xBatch, hiddenStates)
			hiddenStates = forwardResult.H // Update hidden states for the next step
			outputLogits := forwardResult.O // [VocabSize x BatchSize]

			// Calculate loss for this step using standalone softmax (no graph needed)
			probs := SoftmaxStandalone(outputLogits) // [VocabSize x BatchSize]
			stepLoss := 0.0
			numValidInStep := 0
			for j := 0; j < currentBatchSize; j++ { // Iterate through batch items
				targetTokenID := targetTokenIDs[j]
				if targetTokenID == -1 { continue } // Skip invalid/padded targets

				targetProb := probs.Get(targetTokenID, j)
				// Calculate cross-entropy loss: -log(probability of correct token)
				loss := -math.Log(math.Max(targetProb, 1e-9)) // Add epsilon for numerical stability

				if math.IsNaN(loss) || math.IsInf(loss, 0) {
					log.Printf("Warn: NaN/Inf validation loss in Batch %d, Item %d, Step %d. TargetID: %d, Prob: %.4e. Skipping.", batchIndex, j, t, targetTokenID, targetProb)
					continue // Skip this sample's contribution to loss for this step
				}
				numValidInStep++
				stepLoss += loss
			}

			// Accumulate loss for the batch
			if numValidInStep > 0 {
				// Average loss over valid samples in this step? No, sum losses and average at the end.
				batchLoss += stepLoss
				validStepsInBatch += numValidInStep // Count total valid prediction steps
			}
		} // End sequence length loop (t)

		// Accumulate batch loss to total validation loss
		if validStepsInBatch > 0 && !math.IsNaN(batchLoss) && !math.IsInf(batchLoss, 0) {
			totalValidationLoss += batchLoss
			totalValidValidationSteps += validStepsInBatch
		} else if validStepsInBatch > 0 {
			log.Printf("Warn: Invalid total validation batch loss (%.4f) despite %d valid steps in Batch %d.", batchLoss, validStepsInBatch, batchIndex)
		}
	} // End batch loop

	// Calculate average validation loss
	avgValidationLoss := 0.0
	if totalValidValidationSteps > 0 {
		avgValidationLoss = totalValidationLoss / float64(totalValidValidationSteps)
	} else {
		log.Println("Warning: Validation completed with zero valid steps across all batches.")
		return 0.0, nil // Or return an error? Returning 0.0 for now.
	}

	duration := time.Since(startTime)
	log.Printf("Status: Validation loss calculation complete. Avg Loss: %.4f, Duration: %s", avgValidationLoss, duration)
	return avgValidationLoss, nil
}


//======================================================================
// --- Training Loop ---
//======================================================================
// trainGRUModel uses global bpeActualVocabSize
func trainGRUModel(startEpoch int) error {
	if model == nil || solver == nil { return errors.New("training called but model or solver is not initialized") }
	if bpeActualVocabSize <= 0 { return errors.New("training called but BPE vocab size is zero") }

	log.Printf("Status: starting from epoch %d...", startEpoch)
	log.Println("\n--- Training model ---")

	totalBatches := len(batches)
	if totalBatches == 0 { return errors.New("no batches found for training") }

	log.Printf("Starting training: %d total epochs configured, %d batches/epoch, Batch Size: %d, Embedding Dim: %d...", flagEpochs, totalBatches, batchSize, embeddingDimension)

	for epoch := startEpoch; epoch < flagEpochs; epoch++ {
		log.Printf("\nStatus: Starting Epoch %d/%d", epoch+1, flagEpochs)
		epochStartTime := time.Now()
		cumulativeEpochLoss := 0.0
		totalValidStepsInEpoch := 0

		// Shuffle batches at the beginning of each epoch
		rand.Shuffle(len(batches), func(i, j int) {
			batches[i], batches[j] = batches[j], batches[i]
		})

		progressInterval := totalBatches / 20 // Log progress roughly 20 times per epoch
		if progressInterval == 0 { progressInterval = 1 }

		for batchIndex, batch := range batches {
			currentBatchSize := len(batch)
			if currentBatchSize == 0 { continue } // Should not happen if prep is correct

			g := NewGraph(true) // Enable backpropagation
			var hiddenStates [][]*Mat = nil // Reset hidden state for each new batch sequence

			batchLoss := 0.0
			validStepsInBatch := 0 // Total valid (non-padded) steps in this batch sequence

			// --- Process sequence ---
			for t := 0; t < seqLength; t++ {
				inputTokenIDs := make([]int, currentBatchSize)
				targetTokenIDs := make([]int, currentBatchSize)
				hasValidTargetInStep := false

				// Prepare input and target IDs for this time step
				for i := 0; i < currentBatchSize; i++ {
					if t < len(batch[i].Input) && t < len(batch[i].Target) {
						if batch[i].Input[t] >= 0 && batch[i].Input[t] < bpeActualVocabSize {
							inputTokenIDs[i] = batch[i].Input[t]
						} else { inputTokenIDs[i] = -1 } // Use -1 for padding/invalid
						if batch[i].Target[t] >= 0 && batch[i].Target[t] < bpeActualVocabSize {
							targetTokenIDs[i] = batch[i].Target[t]
							hasValidTargetInStep = true
						} else { targetTokenIDs[i] = -1 }
					} else {
						inputTokenIDs[i] = -1; targetTokenIDs[i] = -1
					}
				}

				// If no valid targets in this step, can technically skip, but forward pass maintains state.
				// So, we perform the forward pass regardless, but only calculate loss if targets exist.
				// if !hasValidTargetInStep { continue } // Let's keep the forward pass for state consistency

				// Forward pass for this time step
				xBatch := g.Lookup(model["WE"], inputTokenIDs)
				forwardResult := ForwardMoEGRU(g, model, hiddenSizes, numExperts, xBatch, hiddenStates)
				hiddenStates = forwardResult.H // Update hidden states for the next step
				outputLogits := forwardResult.O // [VocabSize x BatchSize]

				// --- Calculate Loss & Gradients for this step (if valid targets exist) ---
				if hasValidTargetInStep {
					// Need probabilities to calculate loss and gradients w.r.t. logits
					// SoftmaxStandalone calculates probs without adding to graph
					probs := SoftmaxStandalone(outputLogits)
					stepLoss := 0.0
					// Gradient of Loss w.r.t. logits (dL/dLogits = Probs - OneHotTargets)
					dLdLogits := NewMat(bpeActualVocabSize, currentBatchSize) // Zero initialized
					numValidInStep := 0

					for j := 0; j < currentBatchSize; j++ { // Iterate through batch items
						targetTokenID := targetTokenIDs[j]
						if targetTokenID == -1 { continue } // Skip invalid/padded targets

						targetProb := probs.Get(targetTokenID, j)
						loss := -math.Log(math.Max(targetProb, 1e-9)) // Cross-entropy loss

						if math.IsNaN(loss) || math.IsInf(loss, 0) {
							log.Printf("Warn: NaN/Inf training loss Ep %d, Batch %d, Item %d, Step %d. TargetID: %d, Prob: %.4e. Skipping item.", epoch+1, batchIndex, j, t, targetTokenID, targetProb)
							// Zero out gradients for this item if loss is invalid?
							// For now, just skip adding its loss and gradient contribution.
							continue
						}
						numValidInStep++
						stepLoss += loss

						// Calculate dL/dLogits = Probs - OneHotTarget
						// Iterate through vocab for this batch item
						for i := 0; i < bpeActualVocabSize; i++ {
							delta := probs.Get(i, j) // Probability of class i
							if i == targetTokenID {
								delta -= 1.0 // Subtract 1 for the target class
							}
							// Set the gradient, ensuring it's not NaN/Inf
							if !math.IsNaN(delta) && !math.IsInf(delta, 0) {
								dLdLogits.Set(i, j, delta)
							} else {
								dLdLogits.Set(i, j, 0.0) // Set to zero if invalid
							}
						}
					} // End batch item loop (j)

					// Accumulate loss and propagate gradients if valid steps occurred
					if numValidInStep > 0 {
						batchLoss += stepLoss
						validStepsInBatch += numValidInStep

						// Normalize gradient by the number of valid samples in the step
						scaleFactor := 1.0 / float64(numValidInStep)
						// Apply the calculated dL/dLogits gradient to the outputLogits matrix's Dw
						// --- Potential Chunking Start (Apply Logit Grad) ---
						for j := 0; j < currentBatchSize; j++ {
							if targetTokenIDs[j] != -1 { // Only apply grad if target was valid
								for i := 0; i < bpeActualVocabSize; i += defaultChunkSize {
									end_i := i + defaultChunkSize
									if end_i > bpeActualVocabSize { end_i = bpeActualVocabSize }
									for row := i; row < end_i; row++ {
										grad_ij := dLdLogits.Get(row, j)
										// Check NaN/Inf again before accumulating
										if !math.IsNaN(grad_ij) && !math.IsInf(grad_ij, 0) {
											outputLogits.Dw[row*currentBatchSize+j] += grad_ij * scaleFactor
										}
									}
								}
							}
						}
						// --- Potential Chunking End (Apply Logit Grad) ---
					}
				} // End if hasValidTargetInStep
			} // End sequence length loop (t)

			// --- After processing the sequence for the batch ---
			if validStepsInBatch > 0 && !math.IsNaN(batchLoss) && !math.IsInf(batchLoss, 0) {
				// 1. Perform backward pass to compute gradients for all parameters
				g.Backward()

				// 2. Gradient Clipping (Optional but recommended)
				params := GetModelParameters(model)
				var gradNormSq float64 = 0
				// --- Potential Chunking Start (Grad Norm Calc) ---
				for _, p := range params {
					if p == nil { continue }
					dw := p.Dw
					pLen := len(dw)
					for i := 0; i < pLen; i += defaultChunkSize {
						end_i := i + defaultChunkSize
						if end_i > pLen { end_i = pLen }
						normChunkSq := 0.0
						for k := i; k < end_i; k++ {
							dwVal := dw[k]
							if !math.IsNaN(dwVal) && !math.IsInf(dwVal, 0) {
								normChunkSq += dwVal * dwVal
							}
						}
						gradNormSq += normChunkSq
					}
				}
				// --- Potential Chunking End (Grad Norm Calc) ---

				if !math.IsNaN(gradNormSq) && !math.IsInf(gradNormSq, 0) && gradNormSq > 0 {
					gradNorm := math.Sqrt(gradNormSq)
					if gradNorm > flagGradientClipValue {
						scale := flagGradientClipValue / (gradNorm + 1e-7) // Add epsilon for stability
						// --- Potential Chunking Start (Grad Clipping) ---
						for _, p := range params {
							if p == nil { continue }
							dw := p.Dw
							pLen := len(dw)
							for i := 0; i < pLen; i += defaultChunkSize {
								end_i := i + defaultChunkSize
								if end_i > pLen { end_i = pLen }
								for k := i; k < end_i; k++ {
									// Check again before scaling
									if !math.IsNaN(dw[k]) && !math.IsInf(dw[k], 0) {
										dw[k] *= scale
									} else {
										dw[k] = 0 // Zero out invalid gradients found during clipping
									}
								}
							}
						}
						// --- Potential Chunking End (Grad Clipping) ---
					}
					// 3. Update parameters using the optimizer
					solver.OptimizedStep(model) // Use the optimized step (handles parallel/chunking)

				} else {
					log.Printf("Warn: Grad norm invalid (sqrt(%.4f)) or zero Ep %d Batch %d. Zeroing grads and skipping optimizer step.", gradNormSq, epoch+1, batchIndex)
					ZeroModelGrads(model) // Zero grads manually if skipping optimizer step
				}


				// Accumulate loss for epoch average calculation
				cumulativeEpochLoss += batchLoss
				totalValidStepsInEpoch += validStepsInBatch

			} else if validStepsInBatch > 0 {
				// Loss was NaN/Inf, but we had steps. Should not happen if item skip logic is correct.
				log.Printf("Warn: Invalid batch loss (%.4f) despite %d valid steps Ep %d Batch %d. Zeroing grads.", batchLoss, validStepsInBatch, epoch+1, batchIndex)
				ZeroModelGrads(model) // Zero grads as state might be corrupted
			} else {
				// No valid steps in the batch, no loss, no backward pass needed.
				// Gradients should already be zero from previous optimizer step.
			}


			// Progress Logging
			if (batchIndex+1)%progressInterval == 0 || batchIndex == totalBatches-1 {
				doneCount := batchIndex + 1
				percentage := float64(doneCount) / float64(totalBatches) * 100
				barLength := 20
				filledLength := int(percentage / 100 * float64(barLength))
				// Ensure filledLength is within bounds
				if filledLength > barLength { filledLength = barLength }
				if filledLength < 0 { filledLength = 0 }
				bar := strings.Repeat("=", filledLength) + strings.Repeat("-", barLength-filledLength)
				currentAvgLoss := 0.0
				if totalValidStepsInEpoch > 0 {
					currentAvgLoss = cumulativeEpochLoss / float64(totalValidStepsInEpoch)
				}
				fmt.Printf("\rEpoch %d/%d [%s] %d/%d (%.1f%%) Avg Loss: %.4f",
					epoch+1, flagEpochs, bar, doneCount, totalBatches, percentage, currentAvgLoss)
			}
		} // End batch loop

		// --- End of Epoch ---
		fmt.Println() // Newline after progress bar
		avgEpochLoss := 0.0
		if totalValidStepsInEpoch > 0 {
			avgEpochLoss = cumulativeEpochLoss / float64(totalValidStepsInEpoch)
		} else {
			log.Printf("Warning: Epoch %d completed with zero valid training steps.", epoch+1)
		}
		epochDuration := time.Since(epochStartTime)
		log.Printf("Epoch: %d/%d, Average Training Step Loss: %.4f, Duration: %s",
			epoch+1, flagEpochs, avgEpochLoss, epochDuration)

		// Validation Loss Calculation
		if len(validationBatches) > 0 {
			validationLoss, valErr := calculateValidationLoss(model, validationBatches)
			if valErr != nil {
				log.Printf("Error calculating validation loss for epoch %d: %v", epoch+1, valErr)
			} else {
				log.Printf("Epoch: %d/%d, Validation Loss: %.4f", epoch+1, flagEpochs, validationLoss)
			}
		} else {
			log.Printf("Epoch: %d/%d, No validation data provided.", epoch+1, flagEpochs)
		}

		// Save Checkpoint
		checkpointFilename := fmt.Sprintf("checkpoint_epoch_%d.gob", epoch) // Use epoch index (0-based)
		checkpointFilepath := filepath.Join(CheckpointDir, checkpointFilename)
		err := saveCheckpoint(epoch, model, solver, checkpointFilepath)
		if err != nil {
			// Log error but continue training if possible
			log.Printf("Error saving checkpoint for epoch %d: %v", epoch, err)
		}
	} // End epoch loop

	log.Println("--- Training Complete ---")
	log.Println("Status: Training finished.")
	trainingComplete = true // Mark model as ready
	return nil
}


//======================================================================
// --- Conversational Response Generation ---
//======================================================================
// generateResponse takes the BPE instance as an argument
func generateResponse(bpeInstance *BPE, inputText string, maxLength int) (string, error) {
	if !trainingComplete || bpeInstance == nil || model == nil {
		return "Sorry, the model isn't trained or loaded yet.", nil
	}
	if numExperts <= 0 { return "Error: Model config issue (numExperts invalid).", errors.New("numExperts invalid") }
	if _, ok := model["WE"]; !ok { return "Error: Model config issue (WE embedding missing).", errors.New("WE missing") }
	if bpeActualVocabSize <= 0 { return "Error: BPE tokenizer not initialized (vocab size 0).", errors.New("BPE vocab size 0") }

	g := NewGraph(false) // No backprop needed for generation
	var hiddenStates [][]*Mat = nil // Start with nil hidden state

	// Define special tokens and their IDs
	userToken := "[USER]"; botToken := "[BOT]"; eosToken := "[EOS]"
	userTokenID, hasUser := bpeInstance.specialTokensMap[userToken]
	botTokenID, hasBot := bpeInstance.specialTokensMap[botToken]
	eosTokenID, hasEOS := bpeInstance.specialTokensMap[eosToken]
	unkTokenID, hasUnk := bpeInstance.specialTokensMap["[UNK]"] // Needed for invalid prompt tokens

	// --- Prepare Prompt ---
	// Format: [USER] user input text [BOT]
	promptText := fmt.Sprintf("%s %s %s", userToken, inputText, botToken)
	promptIDs := bpeInstance.Encode(promptText)

	// Filter invalid IDs from prompt (-1 or out of bounds) before feeding to model
	validPromptIDsForPriming := []int{}
	for _, id := range promptIDs {
		if id >= 0 && id < bpeActualVocabSize {
			validPromptIDsForPriming = append(validPromptIDsForPriming, id)
		} else {
			log.Printf("Warning: Invalid token ID %d in prompt, treating as UNK/skipping.", id)
			if hasUnk {
				validPromptIDsForPriming = append(validPromptIDsForPriming, unkTokenID)
			} // else: skip if no UNK defined
		}
	}

	if len(validPromptIDsForPriming) == 0 {
		log.Println("Warning: No valid tokens after encoding prompt. Cannot prime the model.")
		return "I couldn't process that input.", nil
	}

	// --- Prime the Model ---
	// Feed the prompt sequence through the model to set the hidden state
	currentTokenID := -1 // Will hold the last valid token ID from the prompt
	for _, tokenID := range validPromptIDsForPriming {
		// Note: Lookup handles -1 internally now, but we filtered them above
		// If we kept -1, Lookup would produce zeros, which might be okay.
		x := g.Lookup(model["WE"], []int{tokenID}) // Batch size of 1
		forwardResult := ForwardMoEGRU(g, model, hiddenSizes, numExperts, x, hiddenStates)
		hiddenStates = forwardResult.H // Update hidden state
		currentTokenID = tokenID       // Keep track of the last token fed
	}

	// Ensure we have a valid starting token ID for generation
	if currentTokenID == -1 {
		log.Println("Error: Failed to set currentTokenId during priming. Using BOT token fallback.")
		if hasBot {
			currentTokenID = botTokenID
		} else {
			return "Error processing input prompt (priming failed).", errors.New("failed to set currentTokenId during priming, no BOT fallback")
		}
	}

	// --- Generate Response ---
	generatedResponseIDs := []int{}
	for t := 0; t < maxLength; t++ {
		// Ensure current token ID is valid before lookup
		if currentTokenID < 0 || currentTokenID >= bpeActualVocabSize {
			log.Printf("Error: Invalid currentTokenId (%d) at start of gen step %d. Stopping.", currentTokenID, t)
			break
		}

		// Forward pass for the current token
		x := g.Lookup(model["WE"], []int{currentTokenID}) // Batch size 1
		forwardResult := ForwardMoEGRU(g, model, hiddenSizes, numExperts, x, hiddenStates)
		hiddenStates = forwardResult.H    // Update hidden states
		outputLogits := forwardResult.O // [VocabSize x 1]

		// Get probabilities from logits
		probs := SoftmaxStandalone(outputLogits)

		// --- Sampling ---
		// Sample the next token ID based on probabilities
		sample := rand.Float64() // Random number [0.0, 1.0)
		cumulativeProb := 0.0
		nextTokenID := -1

		// Check for and handle potential NaN/Inf in probabilities
		probSum := 0.0
		validProbs := true
		for i := 0; i < probs.N; i++ {
			probVal := probs.Get(i, 0)
			if math.IsNaN(probVal) || math.IsInf(probVal, 0) {
				probs.Set(i, 0, 0.0) // Zero out invalid probability
				validProbs = false
			} else {
				probSum += probVal
			}
		}

		// If probabilities were invalid or didn't sum to ~1, handle it
		if !validProbs || math.Abs(probSum-1.0) > 1e-5 {
			log.Printf("Warning: Probs invalid or sum %.5f != 1.0 at step %d. Renormalizing/Uniform sampling.", probSum, t)
			// Option 1: Uniform sampling if sum is zero
			if probSum <= 1e-9 {
				nextTokenID = rand.Intn(bpeActualVocabSize)
				goto EndSampling // Jump past cumulative sampling
			}
			// Option 2: Renormalize
			renormFactor := 1.0 / probSum
			cumulativeProb = 0.0
			for i := 0; i < probs.N; i++ {
				renormalizedProb := probs.Get(i, 0) * renormFactor
				probs.Set(i, 0, renormalizedProb) // Update matrix with renormalized prob
				cumulativeProb += renormalizedProb
				if sample < cumulativeProb && nextTokenID == -1 { // Take the first bin the sample falls into
					nextTokenID = i
				}
			}
			// Fallback if something went wrong with renormalization/sampling
			if nextTokenID == -1 { nextTokenID = bpeActualVocabSize - 1 }

		} else {
			// Standard cumulative probability sampling
			for i := 0; i < bpeActualVocabSize; i++ {
				cumulativeProb += probs.Get(i, 0)
				if sample < cumulativeProb {
					nextTokenID = i
					break
				}
			}
			// Fallback if sample > cumulativeProb (shouldn't happen with valid probs summing to 1)
			if nextTokenID == -1 { nextTokenID = bpeActualVocabSize - 1 }
		}

	EndSampling:

		// --- Check for End Tokens ---
		// Stop if EOS is generated (and defined)
		if hasEOS && nextTokenID == eosTokenID { break }
		// Stop if USER token is generated (model trying to speak as user)
		if hasUser && nextTokenID == userTokenID { break }
		// Stop if BOT token is generated immediately (often indicates confusion)
		// if t == 0 && hasBot && nextTokenID == botTokenID { break } // Optional stricter check

		// Ensure the sampled token ID is valid before adding and continuing
		if nextTokenID < 0 || nextTokenID >= bpeActualVocabSize {
			log.Printf("Error: Sampled invalid token ID %d step %d. Stopping.", nextTokenID, t)
			break
		}

		// Add the generated token to the response sequence
		generatedResponseIDs = append(generatedResponseIDs, nextTokenID)
		// The generated token becomes the input for the next time step
		currentTokenID = nextTokenID

	} // End generation loop (t)

	// --- Decode Response ---
	if len(generatedResponseIDs) == 0 {
		return "...", nil // Return placeholder if nothing was generated
	}
	decodedString := bpeInstance.Decode(generatedResponseIDs)

	// Optional: Clean up potential leading BOT token if it was accidentally included
	if hasBot && len(generatedResponseIDs) > 0 && generatedResponseIDs[0] == botTokenID {
		botTokenString := ""
		if botTokenID >= 0 && botTokenID < len(bpeInstance.vocabArray) {
			botTokenString = bpeInstance.vocabArray[botTokenID]
			// Check if decoded string *starts* with the bot token string (decoding might merge things)
			if strings.HasPrefix(decodedString, botTokenString) {
				decodedString = strings.TrimPrefix(decodedString, botTokenString)
				decodedString = strings.TrimSpace(decodedString) // Trim space left by prefix removal
			}
		}
	}

	finalResponse := strings.TrimSpace(decodedString)
	if finalResponse == "" {
		finalResponse = "..." // Fallback if decoding resulted in empty string
	}

	return finalResponse, nil
}


//======================================================================
// --- Main Execution & Chat Interface ---
//======================================================================

func main() {
	rand.Seed(time.Now().UnixNano())
	log.SetFlags(log.LstdFlags | log.Lshortfile) // Include file/line number

	// --- Define Flags ---
	flag.StringVar(&flagMode, "mode", "chat", "Execution mode: 'bpe-train', 'train', 'chat'")
	// BPE related flags
	flag.IntVar(&flagBPEVocabSize, "bpe-vocab-size", 850, "Target vocabulary size for BPE training (used in bpe-train mode)")
	flag.StringVar(&flagBPEData, "bpe-data", "", "Path to corpus file for BPE training (used in bpe-train mode)")
	flag.StringVar(&flagBPEPath, "bpe-path", "bpe_tokenizer.gob", "Path to load/save BPE tokenizer state file (used in train/chat modes for load, bpe-train for save if -bpe-output not set)")
	flag.StringVar(&flagBPEOutputPath, "bpe-output", "", "Specific path to save trained BPE state (overrides -bpe-path for output in bpe-train mode)")
	// Model architecture/hyperparameters
	flag.IntVar(&flagEmbeddingDimension, "embedding-dim", 96, "Dimension for token embeddings")
	flag.IntVar(&flagGRUHiddenSize, "gru-hidden-size", 96, "Hidden size for GRU layers")
	flag.IntVar(&flagGRULayers, "gru-layers", 2, "Number of GRU layers")
	flag.IntVar(&flagNumExperts, "num-experts", 6, "Number of experts in MoE layers")
	flag.IntVar(&flagTrainSeqLength, "seq-length", 80, "Sequence length for training")
	flag.IntVar(&flagBatchSize, "batch-size", 16, "Batch size for training")
	flag.IntVar(&flagEpochs, "epochs", 5, "Number of training epochs (used in train mode)")
	flag.IntVar(&flagMaxResponseLength, "max-response", 260, "Maximum number of tokens to generate in response (used in chat mode)")
	flag.Float64Var(&flagLearningRate, "lr", 0.001, "Learning rate for AdamW optimizer")
	flag.Float64Var(&flagWeightDecay, "wd", 0.01, "Weight decay for AdamW optimizer")
	flag.Float64Var(&flagEpsilonRMSNorm, "eps-rmsnorm", 1e-5, "Epsilon for RMSNorm stability")
	flag.Float64Var(&flagEpsilonAdamW, "eps-adamw", 1e-8, "Epsilon for AdamW optimizer stability")
	flag.Float64Var(&flagGradientClipValue, "grad-clip", 5.0, "Gradient clipping value")
	// Data and Checkpoint flags for train/chat modes
	flag.StringVar(&flagModelData, "model-data", "", "Path to the data file for model training (required for train mode)")
	flag.StringVar(&flagValData, "validation-data", "", "Path to data for validation loss calculation (optional for train mode)")
	flag.StringVar(&flagCheckpoint, "checkpoint", "", "Path to model checkpoint file (.gob) to load (required for chat, optional for train)")

	// --- Parse Flags ---
	flag.Parse()

	// --- Handle BPE Training Mode Separately ---
	if flagMode == "bpe-train" {
		// Determine output path
		if flagBPEOutputPath == "" {
			if flagBPEPath == "" {
				log.Fatal("FATAL: In bpe-train mode, either --bpe-output or --bpe-path must be specified for saving the tokenizer.")
			}
			flagBPEOutputPath = flagBPEPath // Use --bpe-path as output if --bpe-output isn't set
			log.Printf("Info: --bpe-output not set, using --bpe-path ('%s') for saving.", flagBPEOutputPath)
		}
		err := handleBPETraining()
		if err != nil {
			log.Fatalf("FATAL: BPE Training failed: %v", err)
		}
		log.Println("BPE Training finished. Exiting.")
		os.Exit(0)
	}

	// --- Setup for Train or Chat Mode ---
	log.Printf("Status: Running in '%s' mode.", flagMode)

	// Validate flags for train/chat modes
	if flagMode != "train" && flagMode != "chat" {
		log.Fatalf("FATAL: Invalid mode '%s'. Must be 'bpe-train', 'train', or 'chat'.", flagMode)
	}
	if flagBPEPath == "" {
		log.Fatal("FATAL: --bpe-path flag is required for 'train' and 'chat' modes to load the tokenizer.")
	}
	if flagMode == "train" && flagModelData == "" {
		log.Fatal("FATAL: --model-data flag is required for 'train' mode.")
	}
	if flagMode == "chat" && flagCheckpoint == "" {
		log.Fatal("FATAL: --checkpoint flag is required for 'chat' mode.")
	}

	// --- Load BPE Tokenizer ---
	var bpeLoadErr error
	bpe, bpeLoadErr = LoadBPEState(flagBPEPath)
	if bpeLoadErr != nil {
		log.Fatalf("FATAL: Failed to load BPE state from %s: %v", flagBPEPath, bpeLoadErr)
	}
	bpeActualVocabSize = len(bpe.vocabArray)
	if bpeActualVocabSize <= 0 {
		log.Fatalf("FATAL: Loaded BPE tokenizer has zero vocabulary size.")
	}
	log.Printf("BPE tokenizer loaded successfully (Vocab size: %d).", bpeActualVocabSize)

	// --- Assign Global Config Vars (Initial from Flags) ---
	// These might be overridden by checkpoint config later if loading a checkpoint
	embeddingDimension = flagEmbeddingDimension
	hiddenSizes = make([]int, flagGRULayers)
	for i := range hiddenSizes { hiddenSizes[i] = flagGRUHiddenSize }
	numExperts = flagNumExperts
	seqLength = flagTrainSeqLength
	batchSize = flagBatchSize

	log.Println("--- Effective Configuration (Initial / Flags) ---")
	log.Printf("  Mode: %s", flagMode)
	log.Printf("  BPE Path: %s (Actual Vocab: %d)", flagBPEPath, bpeActualVocabSize)
	log.Printf("  EmbeddingDimension: %d", embeddingDimension)
	log.Printf("  GRUHiddenSize: %d", flagGRUHiddenSize)
	log.Printf("  GRULayers: %d", flagGRULayers)
	log.Printf("  NumExperts: %d", numExperts)
	log.Printf("  TrainSeqLength: %d", seqLength)
	log.Printf("  BatchSize: %d", batchSize)
	log.Printf("  Epochs (Target): %d", flagEpochs)
	log.Printf("  MaxResponseLength: %d", flagMaxResponseLength)
	log.Printf("  LearningRate: %.e", flagLearningRate)
	log.Printf("  WeightDecay: %.e", flagWeightDecay)
	log.Printf("  EpsilonRMSNorm: %.e", flagEpsilonRMSNorm)
	log.Printf("  EpsilonAdamW: %.e", flagEpsilonAdamW)
	log.Printf("  GradientClipValue: %.2f", flagGradientClipValue)
	log.Printf("  HiddenSizes Slice: %v", hiddenSizes)
	log.Println("--------------------------------------------------")

	startEpoch := 0
	var modelLoadErr error

	// --- Load Model Checkpoint or Initialize ---
	if flagCheckpoint != "" {
		// Load existing model state
		var loadedSolver *SolverAdamW
		startEpoch, model, loadedSolver, modelLoadErr = loadCheckpoint(flagCheckpoint, bpeActualVocabSize) // Pass expected vocab size
		if modelLoadErr != nil {
			log.Fatalf("FATAL: Failed to load model checkpoint from %s: %v", flagCheckpoint, modelLoadErr)
		}
		solver = loadedSolver // Assign loaded solver

		log.Println("--- Effective Configuration (After Checkpoint Load) ---")
		log.Printf("  Resuming from Epoch: %d", startEpoch)
		log.Printf("  EmbeddingDimension: %d", embeddingDimension) // Updated by loadCheckpoint
		log.Printf("  NumExperts: %d", numExperts)             // Updated by loadCheckpoint
		log.Printf("  TrainSeqLength: %d", seqLength)             // Updated by loadCheckpoint
		log.Printf("  BatchSize: %d", batchSize)             // Updated by loadCheckpoint
		log.Printf("  MaxResponseLength: %d", flagMaxResponseLength) // Updated by loadCheckpoint
		log.Printf("  LearningRate: %.e", solver.LR)              // From loaded solver
		log.Printf("  WeightDecay: %.e", solver.WD)               // From loaded solver
		log.Printf("  EpsilonRMSNorm: %.e", flagEpsilonRMSNorm)    // Updated by loadCheckpoint
		log.Printf("  EpsilonAdamW: %.e", solver.Eps)             // From loaded solver
		log.Printf("  GradientClipValue: %.2f", flagGradientClipValue) // Updated by loadCheckpoint
		log.Printf("  HiddenSizes Slice: %v", hiddenSizes)         // Updated by loadCheckpoint
		log.Println("--------------------------------------------------")

	} else if flagMode == "train" {
		// Initialize new model for training (no checkpoint provided)
		log.Println("No checkpoint specified. Initializing new model for training...")
		// Use bpeActualVocabSize for both embedding and output layer size
		model = InitMoEGRU(bpeActualVocabSize, embeddingDimension, hiddenSizes, bpeActualVocabSize, numExperts)
		solver = NewSolverAdamW(flagLearningRate, 0.9, 0.999, flagEpsilonAdamW, flagWeightDecay)
		startEpoch = 0 // Start from epoch 0

		// Log parameter count for new model
		totalParams := 0
		keys := make([]string, 0, len(model))
		for k := range model { keys = append(keys, k) }
		sort.Strings(keys)
		for _, k := range keys {
			if m := model[k]; m != nil { totalParams += m.N * m.D }
		}
		log.Printf("-------------------------------------")
		log.Printf("Total parameters for new model: %d", totalParams)
		log.Printf("-------------------------------------")
	} else {
		// Chat mode requires a checkpoint, which wasn't provided or failed to load.
		// This case should have been caught by initial flag validation, but double-check.
		log.Fatal("FATAL: Chat mode requires a checkpoint (--checkpoint) but none was loaded.")
	}

	// --- Prepare Validation Data (if path provided, for train mode) ---
	if flagMode == "train" && flagValData != "" {
		log.Println("Preparing validation data...")
		valDataReady, valDataErr := prepareValidationData(flagValData, bpe) // Pass loaded BPE
		if valDataErr != nil {
			log.Printf("Warning: Validation data preparation failed: %v. Proceeding without validation.", valDataErr)
			validationBatches = nil
		} else if !valDataReady || len(validationBatches) == 0 {
			log.Println("Warning: Validation data preparation resulted in no batches. Proceeding without validation.")
			validationBatches = nil
		} else {
			log.Println("Validation data prepared successfully.")
		}
	} else if flagMode == "train" {
		log.Println("Info: No --validation-data path provided. Skipping validation during training.")
	}


	// --- Execute Training (if in train mode) ---
	if flagMode == "train" {
		// Prepare model training data
		log.Println("Preparing model training data...")
		dataReady, dataErr := prepareModelData(flagModelData, bpe) // Pass loaded BPE
		if dataErr != nil { log.Fatalf("FATAL: Model training data preparation failed: %v", dataErr) }
		if !dataReady || len(batches) == 0 { log.Fatalf("FATAL: Model training data preparation resulted in no batches.") }
		log.Println("Model training data prepared successfully.")

		// Start training loop
		if startEpoch < flagEpochs {
			log.Printf("Proceeding with model training from epoch %d up to target epoch %d...", startEpoch, flagEpochs)
			trainErr := trainGRUModel(startEpoch)
			if trainErr != nil { log.Fatalf("FATAL: Model training failed: %v", trainErr) }
			trainingComplete = true // Set flag after successful training
		} else {
			log.Printf("Loaded checkpoint is already at or beyond the target epoch (%d >= %d). No further training needed.", startEpoch, flagEpochs)
			trainingComplete = true // Mark as ready since checkpoint covers training
			// Optional: Run one final validation pass if validation data exists
			if len(validationBatches) > 0 {
				log.Println("Running final validation pass on loaded/trained model...")
				finalValLoss, valErr := calculateValidationLoss(model, validationBatches)
				if valErr != nil { log.Printf("Error during final validation pass: %v", valErr) } else { log.Printf("Final Validation Loss: %.4f", finalValLoss) }
			}
		}
	} else {
		// In chat mode, model was loaded from checkpoint, training is skipped.
		log.Println("Model loaded from checkpoint. Ready for chat.")
		trainingComplete = true // Mark as ready since checkpoint loaded successfully
	}


	// --- Start Chat Interface (if applicable) ---
	if flagMode == "chat" || trainingComplete {
		if model == nil || bpe == nil {
			log.Fatal("FATAL: Model or BPE is nil before starting chat. This indicates an issue in loading or training.")
		}
		log.Println("\nModel ready. Starting chat interface.")
		log.Println("Type 'exit' or 'quit' to end the chat.")

		reader := bufio.NewReader(os.Stdin)
		for {
			fmt.Print("You> ")
			input, _ := reader.ReadString('\n')
			input = strings.TrimSpace(input)
			if strings.ToLower(input) == "exit" || strings.ToLower(input) == "quit" { break }
			if input == "" { continue }

			botResponse, genErr := generateResponse(bpe, input, flagMaxResponseLength) // Pass BPE instance
			if genErr != nil {
				log.Printf("Error during response generation: %v", genErr)
				fmt.Println("Bot: Sorry, an error occurred while generating the response.")
			} else {
				fmt.Printf("Bot: %s\n", botResponse)
			}
		}
		log.Println("\nGoodbye!")
	} else {
		// Should not be reachable if logic is correct, but as a safeguard:
		log.Fatal("FATAL: Reached end of main without being ready for chat or exiting after BPE training.")
	}
}

// --- Helper Functions for Rand --- (Keep as is)
func randi(a, b int) int { if a >= b { return a }; return rand.Intn(b-a) + a }
func randf(a, b float64) float64 { if a >= b { return a }; return rand.Float64()*(b-a) + a }
func randn(mu, std float64) float64 { return rand.NormFloat64()*std + mu }
func stringSliceToIntSlice(strs []string) ([]int, error) { ints := make([]int, len(strs)); var err error; for i, s := range strs { ints[i], err = strconv.Atoi(s); if err != nil { return nil, fmt.Errorf("error converting '%s' to int: %w", s, err) } }; return ints, nil }
