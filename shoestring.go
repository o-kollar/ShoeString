package main

import (
	"bufio"
	"encoding/gob" // Import gob package
	"encoding/json" // Keep for BPE save/load (deprecated part) if needed elsewhere
	"errors"
	"flag"
	"fmt"
	"io/ioutil" // Keep for BPE save/load (deprecated part) and data loading
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
)

// --- Other Constants ---
const (
	CheckpointDir = "checkpoints" // Directory to save checkpoints
)

// Special BPE Tokens
var BpeSpecialTokens = []string{"[USER]", "[BOT]", "[EOS]", "[PAD]", "[UNK]"}

// --- Global Variables ---
var (
	model              map[string]*Mat // Represents the neural network parameters
	solver             *SolverAdamW
	bpe                *BPE // BPE tokenizer instance
	batches            [][]TrainingSample // Store training data in batches
	validationBatches  [][]TrainingSample // Store validation data in batches
	trainingComplete   bool               = false
	bpeActualVocabSize int
	hiddenSizes        []int   // Derived from flags GRULayers and GRUHiddenSize
	seqLength          int     // Corresponds to flagTrainSeqLength
	numExperts         int     // Corresponds to flagNumExperts
	batchSize          int     // Corresponds to flagBatchSize
	embeddingDimension int     // Corresponds to flagEmbeddingDimension
	// Note: LearningRate etc. are used directly via their flag variables in the relevant functions
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
// *** BPE Tokenizer Implementation *** (Keep BPE struct and methods as is)
//======================================================================
type MergeInfo struct {
	Rank          int      `json:"rank"` // Keep json tags for deprecated Save/Load
	MergedTokenID int      `json:"mergedTokenId"`
	Pair          []string `json:"-"` // derived, not saved directly by key
	Result        string   `json:"-"` // derived
	ID            int      `json:"-"` // derived
}

type BPESavedState struct {
	SpecialTokens []string             `json:"specialTokens"` // Keep json tags for deprecated Save/Load
	VocabArray    []string             `json:"vocabArray"`
	Merges        map[string]MergeInfo `json:"merges"` // Store merges keyed by "token1 token2"
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

// --- BPE Methods (NewBPE, escapeRegex, buildSplitRegex, log, preTokenize, getPairStats, findBestPair, mergeWordTokens, addTokenToVocab, Train, Encode, Decode, GetVocab, GetMerges, GetState, LoadState, Save, Load) ---
// --- No changes needed in the BPE methods themselves for gob checkpointing ---
// --- BPE methods omitted for brevity, they remain the same as in the original code ---
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

// Train now uses the flagBPEVocabSize variable
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
		log.Panic("BPE model not trained.")
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
		log.Panic("BPE model not trained.")
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
	return BPESavedState{
		SpecialTokens: b.specialTokens,
		VocabArray:    b.vocabArray,
		Merges:        b.merges,
	}
}

// LoadState remains the same
func (b *BPE) LoadState(state BPESavedState) error {
	if state.VocabArray == nil || state.Merges == nil || state.SpecialTokens == nil {
		return errors.New("invalid Tokenizer saved state: missing fields")
	}

	b.specialTokens = state.SpecialTokens
	b.vocabArray = state.VocabArray
	b.vocabMap = make(map[string]int, len(b.vocabArray))
	for id, token := range b.vocabArray {
		b.vocabMap[token] = id
	}

	b.merges = state.Merges // Load the map directly

	b.specialTokensMap = make(map[string]int)
	for _, token := range b.specialTokens {
		if id, exists := b.vocabMap[token]; exists {
			b.specialTokensMap[token] = id
		} else {
			log.Printf("Warning: Loaded special token '%s' not in vocab.", token)
			// Optionally add it? Or error? For now, just warn.
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

	b.log("Tokenizer state loaded.")
	return nil
}

// Save (deprecated, use GetState for checkpointing)
func (b *BPE) Save(path string) error {
	state := b.GetState()
	// Use JSON for this specific deprecated save function
	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal BPE state: %w", err)
	}
	err = ioutil.WriteFile(path, data, 0644)
	if err != nil {
		return fmt.Errorf("failed to write BPE state to file %s: %w", path, err)
	}
	b.log(fmt.Sprintf("Tokenizer model saved to %s", path))
	return nil
}

// Load (deprecated, use LoadState for checkpointing)
func (b *BPE) Load(path string) error {
	// Use JSON for this specific deprecated load function
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return fmt.Errorf("failed to read Tokenizer state from file %s: %w", path, err)
	}
	var state BPESavedState
	err = json.Unmarshal(data, &state)
	if err != nil {
		return fmt.Errorf("failed to unmarshal Tokenizer state: %w", err)
	}
	err = b.LoadState(state)
	if err == nil {
		b.log(fmt.Sprintf("BPE model loaded from %s.", path))
	}
	return err
}


//======================================================================
// *** R Library (Matrix Ops, Graph, Activations, RMSNorm, etc.) *** (Keep as is)
//======================================================================
// --- Matrix Definition ---
type Mat struct {
	N  int
	D  int
	W  []float64
	Dw []float64
}

// --- R Library Methods (NewMat, NewRandMat, Zeros, Get, Set, GetCol, ZeroGrads, Clone, Graph, NewGraph, Backward, addBackward, Tanh, Sigmoid, Relu, Gelu, Add, Mul, Eltmul, AddBroadcastCol, Ones, OneMinus, Lookup, CombineExperts, RMSNorm, Softmax, SoftmaxStandalone, StackCols) ---
// --- No changes needed in the R library methods themselves for gob checkpointing ---
// --- R Library methods omitted for brevity, they remain the same as in the original code ---
func NewMat(n, d int) *Mat {
	assert(n >= 0 && d >= 0, "Matrix dimensions must be non-negative")
	if n*d == 0 { // Handle case where either n or d is 0
		return &Mat{N: n, D: d, W: []float64{}, Dw: []float64{}}
	}
	w := make([]float64, n*d)
	dw := make([]float64, n*d)
	return &Mat{N: n, D: d, W: w, Dw: dw}
}
func NewRandMat(n, d int, mu, stddev float64) *Mat {
	m := NewMat(n, d)
	for i := range m.W {
		m.W[i] = rand.NormFloat64()*stddev + mu
	}
	return m
}
func Zeros(n int) []float64 {
	if n <= 0 {
		return []float64{}
	}
	return make([]float64, n)
}
func (m *Mat) Get(row, col int) float64 {
	assert(row >= 0 && row < m.N && col >= 0 && col < m.D, fmt.Sprintf("Mat.Get index (%d,%d) out of bounds for %dx%d matrix", row, col, m.N, m.D))
	ix := row*m.D + col
	return m.W[ix]
}
func (m *Mat) Set(row, col int, v float64) {
	assert(row >= 0 && row < m.N && col >= 0 && col < m.D, fmt.Sprintf("Mat.Set index (%d,%d) out of bounds for %dx%d matrix", row, col, m.N, m.D))
	ix := row*m.D + col
	m.W[ix] = v
}
func (m *Mat) GetCol(col int) *Mat {
	assert(col >= 0 && col < m.D, fmt.Sprintf("Mat.GetCol index %d out of bounds for %dx%d matrix", col, m.N, m.D))
	colMat := NewMat(m.N, 1)
	for i := 0; i < m.N; i++ {
		colMat.W[i] = m.Get(i, col)
	}
	return colMat
}
func (m *Mat) ZeroGrads() {
	for i := range m.Dw {
		m.Dw[i] = 0
	}
}
func (m *Mat) Clone() *Mat {
	newM := NewMat(m.N, m.D)
	copy(newM.W, m.W)
	return newM
}
type Graph struct {
	NeedsBackprop bool
	Backprop      []func()
	mu            sync.Mutex
}
func NewGraph(needsBackprop bool) *Graph {
	return &Graph{
		NeedsBackprop: needsBackprop,
		Backprop:      []func(){},
	}
}
func (g *Graph) Backward() {
	for i := len(g.Backprop) - 1; i >= 0; i-- {
		g.Backprop[i]()
	}
}
func (g *Graph) addBackward(f func()) {
	if g.NeedsBackprop {
		g.mu.Lock()
		g.Backprop = append(g.Backprop, f)
		g.mu.Unlock()
	}
}
const (
	invSqrt2   = 0.7071067811865476
	invSqrt2pi = 0.3989422804014327
)
func applyActivation(g *Graph, m *Mat, activationFn func(float64) float64, derivativeFn func(float64, float64) float64) *Mat {
	out := NewMat(m.N, m.D)
	nTotal := len(m.W)
	for i := 0; i < nTotal; i++ {
		out.W[i] = activationFn(m.W[i])
	}
	if g.NeedsBackprop {
		backward := func() {
			for i := 0; i < nTotal; i++ {
				m.Dw[i] += derivativeFn(m.W[i], out.W[i]) * out.Dw[i]
			}
		}
		g.addBackward(backward)
	}
	return out
}
func (g *Graph) Tanh(m *Mat) *Mat {
	return applyActivation(g, m, math.Tanh, func(m_wi, out_wi float64) float64 {
		return 1.0 - out_wi*out_wi
	})
}
func (g *Graph) Sigmoid(m *Mat) *Mat {
	sigmoid := func(x float64) float64 { return 1.0 / (1.0 + math.Exp(-x)) }
	derivative := func(m_wi, out_wi float64) float64 {
		return out_wi * (1.0 - out_wi)
	}
	return applyActivation(g, m, sigmoid, derivative)
}
func (g *Graph) Relu(m *Mat) *Mat {
	relu := func(x float64) float64 { return math.Max(0, x) }
	derivative := func(m_wi, out_wi float64) float64 {
		if m_wi > 0 { return 1.0 }; return 0.0
	}
	return applyActivation(g, m, relu, derivative)
}
func (g *Graph) Gelu(m *Mat) *Mat {
	geluFunc := func(x float64) float64 {
		return 0.5 * x * (1.0 + math.Erf(x*invSqrt2))
	}
	geluDerivative := func(x, gelu_x float64) float64 {
		phi_x := invSqrt2pi * math.Exp(-0.5*x*x)
		var phi_cap_x float64
		if math.Abs(x) < 1e-9 { phi_cap_x = 0.5 } else { phi_cap_x = gelu_x / x }
		derivative := phi_cap_x + x*phi_x
		return derivative
	}
	return applyActivation(g, m, geluFunc, geluDerivative)
}
func (g *Graph) Add(m1, m2 *Mat) *Mat {
	assert(m1.N == m2.N && m1.D == m2.D, fmt.Sprintf("Add: Matrices must have the same size. m1: %dx%d, m2: %dx%d", m1.N, m1.D, m2.N, m2.D))
	out := NewMat(m1.N, m1.D)
	for i := range m1.W { out.W[i] = m1.W[i] + m2.W[i] }
	if g.NeedsBackprop {
		backward := func() {
			for i := range m1.W { m1.Dw[i] += out.Dw[i]; m2.Dw[i] += out.Dw[i] }
		}
		g.addBackward(backward)
	}
	return out
}
func (g *Graph) Mul(m1, m2 *Mat) *Mat {
	assert(m1.D == m2.N, fmt.Sprintf("Mul: Matrix dimensions misaligned. m1: %dx%d, m2: %dx%d", m1.N, m1.D, m2.N, m2.D))
	n := m1.N; k := m1.D; batchSizeOut := m2.D
	out := NewMat(n, batchSizeOut)
	for i := 0; i < n; i++ {
		for j := 0; j < batchSizeOut; j++ {
			dot := 0.0
			for l := 0; l < k; l++ { dot += m1.W[i*k+l] * m2.W[l*batchSizeOut+j] }
			out.W[i*batchSizeOut+j] = dot
		}
	}
	if g.NeedsBackprop {
		backward := func() {
			for i := 0; i < n; i++ {
				for j := 0; j < batchSizeOut; j++ {
					gradOut := out.Dw[i*batchSizeOut+j]
					if gradOut == 0 { continue }
					for l := 0; l < k; l++ {
						m1.Dw[i*k+l] += m2.W[l*batchSizeOut+j] * gradOut
						m2.Dw[l*batchSizeOut+j] += m1.W[i*k+l] * gradOut
					}
				}
			}
		}
		g.addBackward(backward)
	}
	return out
}
func (g *Graph) Eltmul(m1, m2 *Mat) *Mat {
	assert(m1.N == m2.N && m1.D == m2.D, fmt.Sprintf("Eltmul: Matrices must have the same size. m1: %dx%d, m2: %dx%d", m1.N, m1.D, m2.N, m2.D))
	out := NewMat(m1.N, m1.D)
	for i := range m1.W { out.W[i] = m1.W[i] * m2.W[i] }
	if g.NeedsBackprop {
		backward := func() {
			for i := range m1.W { m1.Dw[i] += m2.W[i] * out.Dw[i]; m2.Dw[i] += m1.W[i] * out.Dw[i] }
		}
		g.addBackward(backward)
	}
	return out
}
func (g *Graph) AddBroadcastCol(m1 *Mat, m2Col *Mat) *Mat {
	assert(m1.N == m2Col.N, fmt.Sprintf("AddBroadcastCol: Row dimension mismatch. m1: %dx%d, m2Col: %dx%d", m1.N, m1.D, m2Col.N, m2Col.D))
	assert(m2Col.D == 1, fmt.Sprintf("AddBroadcastCol: m2Col must be a column vector (D=1), got %dx%d", m2Col.N, m2Col.D))
	n := m1.N; batchSize := m1.D
	out := NewMat(n, batchSize)
	for j := 0; j < batchSize; j++ {
		for i := 0; i < n; i++ { out.W[i*batchSize+j] = m1.W[i*batchSize+j] + m2Col.W[i] }
	}
	if g.NeedsBackprop {
		backward := func() {
			for j := 0; j < batchSize; j++ {
				for i := 0; i < n; i++ {
					gradOut := out.Dw[i*batchSize+j]
					m1.Dw[i*batchSize+j] += gradOut
					m2Col.Dw[i] += gradOut
				}
			}
		}
		g.addBackward(backward)
	}
	return out
}
func (g *Graph) Ones(n, d int) *Mat {
	m := NewMat(n, d)
	for i := range m.W { m.W[i] = 1.0 }
	return m
}
func (g *Graph) OneMinus(m *Mat) *Mat {
	out := NewMat(m.N, m.D)
	for i := range m.W { out.W[i] = 1.0 - m.W[i] }
	if g.NeedsBackprop {
		backward := func() {
			for i := range m.W { m.Dw[i] += -1.0 * out.Dw[i] }
		}
		g.addBackward(backward)
	}
	return out
}
func (g *Graph) Lookup(embeddingMatrix *Mat, tokenIDs []int) *Mat {
	vocabSize := embeddingMatrix.N; embeddingDim := embeddingMatrix.D; batchSize := len(tokenIDs)
	assert(batchSize > 0, "Lookup: tokenIDs slice cannot be empty.")
	out := NewMat(embeddingDim, batchSize)
	validIndices := make([]int, batchSize)
	for j, tokenID := range tokenIDs {
		validIndices[j] = tokenID
		if tokenID < 0 || tokenID >= vocabSize { validIndices[j] = -1; continue } // Handle invalid index
		srcOffset := tokenID * embeddingDim; destCol := j
		for i := 0; i < embeddingDim; i++ { out.W[i*batchSize+destCol] = embeddingMatrix.W[srcOffset+i] }
	}
	if g.NeedsBackprop {
		backward := func() {
			for j := 0; j < batchSize; j++ {
				tokenID := validIndices[j]
				if tokenID == -1 { continue } // Skip backprop for invalid indices
				targetRowOffset := tokenID * embeddingDim; srcCol := j
				for i := 0; i < embeddingDim; i++ {
					grad := out.Dw[i*batchSize+srcCol]
					if !math.IsNaN(grad) && !math.IsInf(grad, 0) { embeddingMatrix.Dw[targetRowOffset+i] += grad }
				}
			}
		}
		g.addBackward(backward)
	}
	return out
}
func (g *Graph) CombineExperts(expertOutputs []*Mat, gatingWeights *Mat) *Mat {
	if len(expertOutputs) == 0 { log.Panic("CombineExperts: expertOutputs slice cannot be empty.") }
	if gatingWeights == nil { log.Panic("CombineExperts: gatingWeights cannot be nil.") }
	numExperts := len(expertOutputs); hiddenSize := expertOutputs[0].N; batchSize := expertOutputs[0].D
	assert(gatingWeights.N == numExperts, fmt.Sprintf("CombineExperts: gatingWeights rows (%d) must match numExperts (%d)", gatingWeights.N, numExperts))
	assert(gatingWeights.D == batchSize, fmt.Sprintf("CombineExperts: gatingWeights cols (%d) must match batch size (%d)", gatingWeights.D, batchSize))
	for e := 0; e < numExperts; e++ {
		assert(expertOutputs[e] != nil, fmt.Sprintf("CombineExperts: expertOutput %d is nil", e))
		assert(expertOutputs[e].N == hiddenSize, fmt.Sprintf("CombineExperts: expertOutput %d rows (%d) must match hiddenSize (%d)", e, expertOutputs[e].N, hiddenSize))
		assert(expertOutputs[e].D == batchSize, fmt.Sprintf("CombineExperts: expertOutput %d cols (%d) must match batch size (%d)", e, expertOutputs[e].D, batchSize))
	}
	out := NewMat(hiddenSize, batchSize)
	for e := 0; e < numExperts; e++ {
		expertOut_e := expertOutputs[e]
		for j := 0; j < batchSize; j++ {
			gateWeight_ej := gatingWeights.Get(e, j)
			if gateWeight_ej == 0 { continue }
			outOffset := j; expertOffset := j
			for i := 0; i < hiddenSize; i++ { out.W[i*batchSize+outOffset] += expertOut_e.W[i*batchSize+expertOffset] * gateWeight_ej }
		}
	}
	if g.NeedsBackprop {
		backward := func() {
			for e := 0; e < numExperts; e++ {
				for j := 0; j < batchSize; j++ {
					gradAccumGating_ej := 0.0
					outOffset := j; expertOffset := j
					for i := 0; i < hiddenSize; i++ {
						gradOut_ij := out.Dw[i*batchSize+outOffset]
						expertVal_eij := expertOutputs[e].W[i*batchSize+expertOffset]
						gradAccumGating_ej += gradOut_ij * expertVal_eij
					}
					gwOffset := e*batchSize + j
					if !math.IsNaN(gradAccumGating_ej) && !math.IsInf(gradAccumGating_ej, 0) { gatingWeights.Dw[gwOffset] += gradAccumGating_ej }
				}
			}
			for e := 0; e < numExperts; e++ {
				expertOutDw_e := expertOutputs[e].Dw
				for j := 0; j < batchSize; j++ {
					gateWeight_ej := gatingWeights.Get(e, j)
					if gateWeight_ej == 0 { continue }
					outOffset := j; expertDwOffset := j
					for i := 0; i < hiddenSize; i++ {
						gradOut_ij := out.Dw[i*batchSize+outOffset]
						gradExpOut_eij := gradOut_ij * gateWeight_ej
						if !math.IsNaN(gradExpOut_eij) && !math.IsInf(gradExpOut_eij, 0) { expertOutDw_e[i*batchSize+expertDwOffset] += gradExpOut_eij }
					}
				}
			}
		}
		g.addBackward(backward)
	}
	return out
}
func (g *Graph) RMSNorm(m, gain *Mat) *Mat {
	assert(gain.N == m.N, fmt.Sprintf("RMSNorm gain rows must match input rows. m: %dx%d, gain: %dx%d", m.N, m.D, gain.N, gain.D))
	assert(gain.D == 1, fmt.Sprintf("RMSNorm gain must be a column vector (D=1). Got %dx%d", gain.N, gain.D))
	n := m.N; batchSize := m.D
	out := NewMat(n, batchSize); rmsPerCol := make([]float64, batchSize); invRMSPerCol := make([]float64, batchSize)
	mNorm := NewMat(n, batchSize)
	for j := 0; j < batchSize; j++ {
		meanSq := 0.0
		for i := 0; i < n; i++ { val := m.Get(i, j); meanSq += val * val }
		meanSq /= float64(n)
		rmsPerCol[j] = math.Sqrt(meanSq + flagEpsilonRMSNorm)
		invRMSPerCol[j] = 1.0 / rmsPerCol[j]
		for i := 0; i < n; i++ {
			normVal := m.Get(i, j) * invRMSPerCol[j]
			mNorm.Set(i, j, normVal)
			out.Set(i, j, normVal*gain.W[i])
		}
	}
	if g.NeedsBackprop {
		backward := func() {
			gainDwTemp := Zeros(n)
			for j := 0; j < batchSize; j++ {
				sumDNormMTimesNegNormM_j := 0.0; dNormM_j := Zeros(n)
				for i := 0; i < n; i++ {
					dOut_ij := out.Dw[i*batchSize+j]; mNorm_ij := mNorm.Get(i, j); gain_i := gain.W[i]
					gainDwTemp[i] += dOut_ij * mNorm_ij
					dNormM_j[i] = dOut_ij * gain_i
					sumDNormMTimesNegNormM_j += dNormM_j[i] * (-mNorm_ij)
				}
				dRMS_j := sumDNormMTimesNegNormM_j * invRMSPerCol[j]
				dMeanSq_j := dRMS_j * 0.5 * invRMSPerCol[j]
				for i := 0; i < n; i++ {
					gradMDirect := dNormM_j[i] * invRMSPerCol[j]
					gradMIndirect := dMeanSq_j * (2.0 * m.Get(i, j) / float64(n))
					m.Dw[i*batchSize+j] += gradMDirect + gradMIndirect
				}
			}
			for i := 0; i < n; i++ { gain.Dw[i] += gainDwTemp[i] }
		}
		g.addBackward(backward)
	}
	return out
}
func (g *Graph) Softmax(m *Mat) *Mat {
	n := m.N; batchSize := m.D
	out := NewMat(n, batchSize)
	for j := 0; j < batchSize; j++ {
		maxVal := -math.MaxFloat64
		for i := 0; i < n; i++ { val := m.Get(i, j); if val > maxVal { maxVal = val } }
		sumExp := 0.0; expValsCol := Zeros(n)
		for i := 0; i < n; i++ {
			expVal := math.Exp(m.Get(i, j) - maxVal)
			if math.IsNaN(expVal) || math.IsInf(expVal, 0) { expVal = 0 }
			expValsCol[i] = expVal; sumExp += expVal
		}
		invSumExp := 1.0 / (sumExp + 1e-9)
		if sumExp < 1e-9 {
			invSumExp = 1.0 / float64(n)
			for i := 0; i < n; i++ { out.Set(i, j, invSumExp) }
		} else {
			for i := 0; i < n; i++ { out.Set(i, j, expValsCol[i]*invSumExp) }
		}
	}
	if g.NeedsBackprop {
		backward := func() {
			for j := 0; j < batchSize; j++ {
				dL_dOutput_j := Zeros(n); probs_j := Zeros(n)
				for i := 0; i < n; i++ { dL_dOutput_j[i] = out.Dw[i*batchSize+j]; probs_j[i] = out.W[i*batchSize+j] }
				dotProd := 0.0
				for k := 0; k < n; k++ {
					if !math.IsNaN(dL_dOutput_j[k]) && !math.IsInf(dL_dOutput_j[k], 0) && !math.IsNaN(probs_j[k]) && !math.IsInf(probs_j[k], 0) {
						dotProd += dL_dOutput_j[k] * probs_j[k]
					}
				}
				if math.IsNaN(dotProd) || math.IsInf(dotProd, 0) { dotProd = 0 }
				for i := 0; i < n; i++ {
					prob_i := probs_j[i]; dL_dOutput_i := dL_dOutput_j[i]
					if math.IsNaN(prob_i) || math.IsInf(prob_i, 0) || math.IsNaN(dL_dOutput_i) || math.IsInf(dL_dOutput_i, 0) { continue }
					gradInput_i := prob_i * (dL_dOutput_i - dotProd)
					if !math.IsNaN(gradInput_i) && !math.IsInf(gradInput_i, 0) { m.Dw[i*batchSize+j] += gradInput_i }
				}
			}
		}
		g.addBackward(backward)
	}
	return out
}
func SoftmaxStandalone(m *Mat) *Mat {
	n := m.N; batchSize := m.D
	out := NewMat(n, batchSize)
	for j := 0; j < batchSize; j++ {
		maxVal := -math.MaxFloat64
		for i := 0; i < n; i++ { val := m.Get(i, j); if val > maxVal { maxVal = val } }
		s := 0.0; expValsCol := Zeros(n)
		for i := 0; i < n; i++ {
			expVal := math.Exp(m.Get(i, j) - maxVal)
			if math.IsNaN(expVal) || math.IsInf(expVal, 0) { expVal = 0 }
			expValsCol[i] = expVal; s += expVal
		}
		invS := 1.0 / (s + 1e-9)
		if s < 1e-9 {
			invS = 1.0 / float64(n)
			for i := 0; i < n; i++ { out.Set(i, j, invS) }
		} else {
			for i := 0; i < n; i++ { out.Set(i, j, expValsCol[i]*invS) }
		}
	}
	return out
}
func StackCols(g *Graph, mats []*Mat) *Mat {
	if len(mats) == 0 { log.Panic("stackCols requires a non-empty array of matrices.") }
	n := mats[0].N; numMats := len(mats); dOut := numMats
	for i := 0; i < numMats; i++ {
		assert(mats[i] != nil, fmt.Sprintf("stackCols: Matrix %d is nil.", i))
		assert(mats[i].N == n, fmt.Sprintf("stackCols: Matrix %d has height %d, expected %d.", i, mats[i].N, n))
		assert(mats[i].D == 1, fmt.Sprintf("stackCols: Matrix %d has width %d, expected 1.", i, mats[i].D))
	}
	out := NewMat(n, dOut)
	for j := 0; j < numMats; j++ {
		for i := 0; i < n; i++ { out.W[i*dOut+j] = mats[j].W[i] }
	}
	if g.NeedsBackprop {
		backward := func() {
			for j := 0; j < numMats; j++ {
				for i := 0; i < n; i++ { mats[j].Dw[i] += out.Dw[i*dOut+j] }
			}
		}
		g.addBackward(backward)
	}
	return out
}

//======================================================================
// --- MoE MinGRU Model Definition ---
//======================================================================
// InitMoEGRU remains the same logically
func InitMoEGRU(vocabSize int, embeddingDim int, hiddenSizes []int, outputSize int, numExperts int) map[string]*Mat {
	log.Printf("Initializing model parameters (Experts: %d)...", numExperts)
	model := make(map[string]*Mat)

	initStdDev := func(size int) float64 {
		if size > 0 { return 0.08 } // Keep previous simpler init for now
		return 0.08
	}

	// --- Embedding Layer ---
	log.Printf("Initializing Embedding Layer WE: %d x %d", vocabSize, embeddingDim)
	stdEmbed := 0.02
	model["WE"] = NewRandMat(vocabSize, embeddingDim, 0, stdEmbed)

	// --- GRU Layers ---
	layerInputSize := embeddingDim
	for d, hiddenSize := range hiddenSizes {
		log.Printf("Layer %d: Input Size %d, Hidden Size %d", d, layerInputSize, hiddenSize)
		stdGate := initStdDev(layerInputSize)
		model[fmt.Sprintf("Wg%d", d)] = NewRandMat(numExperts, layerInputSize, 0, stdGate)
		model[fmt.Sprintf("bg%d", d)] = NewMat(numExperts, 1)

		for e := 0; e < numExperts; e++ {
			stdX := initStdDev(layerInputSize); stdH := initStdDev(hiddenSize)
			expertSuffix := fmt.Sprintf("_exp%d", e)
			model[fmt.Sprintf("Wzx%d%s", d, expertSuffix)] = NewRandMat(hiddenSize, layerInputSize, 0, stdX)
			model[fmt.Sprintf("bz%d%s", d, expertSuffix)] = NewMat(hiddenSize, 1)
			model[fmt.Sprintf("Whx%d%s", d, expertSuffix)] = NewRandMat(hiddenSize, layerInputSize, 0, stdX)
			model[fmt.Sprintf("Whh%d%s", d, expertSuffix)] = NewRandMat(hiddenSize, hiddenSize, 0, stdH)
			model[fmt.Sprintf("bh%d%s", d, expertSuffix)] = NewMat(hiddenSize, 1)
		}

		if layerInputSize != hiddenSize {
			log.Printf("  Layer %d: Adding projection %dx%d for residual connection.", d, hiddenSize, layerInputSize)
			stdProj := initStdDev(layerInputSize)
			model[fmt.Sprintf("Wp%d", d)] = NewRandMat(hiddenSize, layerInputSize, 0, stdProj)
			model[fmt.Sprintf("bp%d", d)] = NewMat(hiddenSize, 1)
		} else {
			log.Printf("  Layer %d: Residual connection dimensions match (%dx%d), no projection needed.", d, hiddenSize, layerInputSize)
		}

		log.Printf("  Layer %d: Adding RMSNorm gain parameter g_rms%d (%dx1).", d, d, hiddenSize)
		gRMSKey := fmt.Sprintf("g_rms%d", d)
		model[gRMSKey] = NewMat(hiddenSize, 1)
		for i := range model[gRMSKey].W { model[gRMSKey].W[i] = 1.0 }
		layerInputSize = hiddenSize
	}

	// --- Output Layer ---
	finalHiddenSize := layerInputSize
	if len(hiddenSizes) > 0 { finalHiddenSize = hiddenSizes[len(hiddenSizes)-1] }
	log.Printf("Initializing Output Layer Whd: %d x %d", outputSize, finalHiddenSize)
	stdDec := initStdDev(finalHiddenSize)
	model["Whd"] = NewRandMat(outputSize, finalHiddenSize, 0, stdDec)
	model["bd"] = NewMat(outputSize, 1)

	log.Println("Parameter Keys Initialized:", len(model))
	return model
}

// ForwardResult holds the outputs of the forward pass.
type ForwardResult struct {
	H [][]*Mat // Hidden states per layer, per expert [layer][expert][HiddenSize x BatchSize]
	O *Mat      // Output logits [VocabSize x BatchSize]
}

// ForwardMoEGRU remains the same logically
func ForwardMoEGRU(g *Graph, model map[string]*Mat, hiddenSizes []int, numExperts int, x *Mat, prevHiddenStates [][]*Mat) ForwardResult {
	currentBatchSize := x.D
	needsInit := prevHiddenStates == nil || len(prevHiddenStates) != len(hiddenSizes)
	if !needsInit {
		for dChk := 0; dChk < len(hiddenSizes); dChk++ {
			if len(prevHiddenStates[dChk]) != numExperts { needsInit = true; break }
			if len(prevHiddenStates[dChk]) > 0 {
				if prevHiddenStates[dChk][0] == nil || prevHiddenStates[dChk][0].N != hiddenSizes[dChk] || prevHiddenStates[dChk][0].D != currentBatchSize {
					needsInit = true; break
				}
			} else { needsInit = true; break }
		}
	}
	if needsInit {
		prevHiddenStates = make([][]*Mat, len(hiddenSizes))
		for dInit := 0; dInit < len(hiddenSizes); dInit++ {
			prevHiddenStates[dInit] = make([]*Mat, numExperts)
			for eInit := 0; eInit < numExperts; eInit++ {
				prevHiddenStates[dInit][eInit] = NewMat(hiddenSizes[dInit], currentBatchSize)
			}
		}
	}

	currentHiddenStatesLayers := make([][]*Mat, len(hiddenSizes))
	inputToLayer := x

	for d, hiddenSize := range hiddenSizes {
		layerInputSize := inputToLayer.N
		expertOutputs := make([]*Mat, numExperts)
		currentLayerExpertStates := make([]*Mat, numExperts)
		residualSource := inputToLayer

		wgKey := fmt.Sprintf("Wg%d", d); bgKey := fmt.Sprintf("bg%d", d)
		Wg := model[wgKey]; bg := model[bgKey]
		assert(Wg != nil && bg != nil, fmt.Sprintf("Gating weights %s or %s not found", wgKey, bgKey))
		assert(Wg.D == layerInputSize, fmt.Sprintf("Wg dim mismatch layer %d. Wg.D=%d, layerInputSize=%d", d, Wg.D, layerInputSize))
		gatingLogitsLinear := g.Mul(Wg, inputToLayer)
		gatingLogits := g.AddBroadcastCol(gatingLogitsLinear, bg)
		gatingWeights := g.Softmax(gatingLogits)
		assert(gatingWeights.N == numExperts && gatingWeights.D == currentBatchSize, fmt.Sprintf("Gating weights dim error layer %d", d))

		var wgExperts sync.WaitGroup
		wgExperts.Add(numExperts)
		for e := 0; e < numExperts; e++ {
			go func(expertIdx int) {
				defer wgExperts.Done()
				hPrevExpert := prevHiddenStates[d][expertIdx]
				assert(hPrevExpert.N == hiddenSize && hPrevExpert.D == currentBatchSize, fmt.Sprintf("Prev hidden state dim error layer %d exp %d. hPrev: %dx%d, expected: %dx%d", d, expertIdx, hPrevExpert.N, hPrevExpert.D, hiddenSize, currentBatchSize))
				expertSuffix := fmt.Sprintf("_exp%d", expertIdx)
				wzxKey, bzKey := fmt.Sprintf("Wzx%d%s", d, expertSuffix), fmt.Sprintf("bz%d%s", d, expertSuffix)
				whxKey, whhKey, bhKey := fmt.Sprintf("Whx%d%s", d, expertSuffix), fmt.Sprintf("Whh%d%s", d, expertSuffix), fmt.Sprintf("bh%d%s", d, expertSuffix)
				Wzx_e, bz_e := model[wzxKey], model[bzKey]
				Whx_e, Whh_e, bh_e := model[whxKey], model[whhKey], model[bhKey]
				assert(Wzx_e != nil && bz_e != nil && Whx_e != nil && Whh_e != nil && bh_e != nil, fmt.Sprintf("Missing weights L%d E%d", d, expertIdx))

				zLinear := g.Mul(Wzx_e, inputToLayer)
				z_t_e := g.Sigmoid(g.AddBroadcastCol(zLinear, bz_e))
				termWhx := g.Mul(Whx_e, inputToLayer)
				termWhh := g.Mul(Whh_e, hPrevExpert)
				hCandLinear := g.Add(termWhx, termWhh)
				hCandidate_e := g.Gelu(g.AddBroadcastCol(hCandLinear, bh_e))
				oneMinusZ_e := g.OneMinus(z_t_e)
				term1_e := g.Eltmul(oneMinusZ_e, hPrevExpert)
				term2_e := g.Eltmul(z_t_e, hCandidate_e)
				hNewExpert := g.Add(term1_e, term2_e)
				assert(hNewExpert.N == hiddenSize && hNewExpert.D == currentBatchSize, fmt.Sprintf("h_new_expert dim error L%d E%d", d, expertIdx))
				expertOutputs[expertIdx] = hNewExpert
				currentLayerExpertStates[expertIdx] = hNewExpert
			}(e)
		}
		wgExperts.Wait()

		hNewCombined := g.CombineExperts(expertOutputs, gatingWeights)

		var projectedResidual *Mat
		if layerInputSize == hiddenSize {
			projectedResidual = residualSource
		} else {
			wpKey, bpKey := fmt.Sprintf("Wp%d", d), fmt.Sprintf("bp%d", d)
			Wp, bp := model[wpKey], model[bpKey]
			assert(Wp != nil && bp != nil, fmt.Sprintf("Projection Wp%d or bp%d not found.", d, d))
			projLinear := g.Mul(Wp, residualSource)
			projectedResidual = g.AddBroadcastCol(projLinear, bp)
		}
		assert(projectedResidual.N == hNewCombined.N && projectedResidual.D == hNewCombined.D, "Residual dim mismatch")
		outputWithResidual := g.Add(hNewCombined, projectedResidual)

		gRMSKey := fmt.Sprintf("g_rms%d", d)
		gRMS := model[gRMSKey]
		assert(gRMS != nil && gRMS.N == hiddenSize && gRMS.D == 1, fmt.Sprintf("RMSNorm gain g_rms%d error.", d))
		normalizedOutput := g.RMSNorm(outputWithResidual, gRMS)
		assert(normalizedOutput.N == hiddenSize && normalizedOutput.D == currentBatchSize, fmt.Sprintf("RMSNorm output dim error L%d", d))

		currentHiddenStatesLayers[d] = currentLayerExpertStates
		inputToLayer = normalizedOutput
	}

	lastLayerOutput := inputToLayer
	finalHiddenSize := lastLayerOutput.N
	Whd, bd := model["Whd"], model["bd"]
	assert(Whd != nil && bd != nil, "Output weights Whd or bd not found")
	assert(Whd.D == finalHiddenSize, fmt.Sprintf("Output Whd dim mismatch. Whd.D=%d, finalHiddenSize=%d", Whd.D, finalHiddenSize))
	outputLogitsLinear := g.Mul(Whd, lastLayerOutput)
	outputLogits := g.AddBroadcastCol(outputLogitsLinear, bd)
	assert(outputLogits.N == bpeActualVocabSize && outputLogits.D == currentBatchSize, fmt.Sprintf("Output logits dim error. Got %dx%d, expected %dx%d", outputLogits.N, outputLogits.D, bpeActualVocabSize, currentBatchSize))

	return ForwardResult{H: currentHiddenStatesLayers, O: outputLogits}
}


//======================================================================
// --- Model Parameter Utilities --- (Keep as is)
//======================================================================
func GetModelParameters(model map[string]*Mat) []*Mat {
	params := make([]*Mat, 0, len(model))
	keys := make([]string, 0, len(model))
	for k := range model { keys = append(keys, k) }
	sort.Strings(keys)
	for _, k := range keys { params = append(params, model[k]) }
	return params
}

func ZeroModelGrads(model map[string]*Mat) {
	for _, mat := range model { mat.ZeroGrads() }
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
	M         map[string][]float64 // Momentum - **Must be exported for gob**
	V         map[string][]float64 // Velocity - **Must be exported for gob**
	paramKeys map[string]bool      // Track keys seen (not saved directly)
}

// NewSolverAdamW remains the same
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
		M:         make(map[string][]float64),
		V:         make(map[string][]float64),
		paramKeys: make(map[string]bool),
	}
}

// Step remains the same logically
func (s *SolverAdamW) Step(model map[string]*Mat) {
	s.T++
	t := float64(s.T)
	beta1PowT := math.Pow(s.Beta1, t)
	beta2PowT := math.Pow(s.Beta2, t)
	lrT := s.LR * math.Sqrt(1.0-beta2PowT) / (1.0-beta1PowT)

	keys := make([]string, 0, len(model))
	for k := range model { keys = append(keys, k) }
	sort.Strings(keys)

	for _, k := range keys {
		p := model[k]
		if _, exists := s.paramKeys[k]; !exists {
			s.M[k] = Zeros(len(p.W))
			s.V[k] = Zeros(len(p.W))
			s.paramKeys[k] = true
		}
	}

	for _, k := range keys {
		p := model[k]
		mK, mExists := s.M[k]
		vK, vExists := s.V[k]

		if !mExists || !vExists || len(mK) != len(p.W) || len(vK) != len(p.W) {
			log.Printf("Error: Optimizer state mismatch for key %s. Reinitializing.", k)
			s.M[k] = Zeros(len(p.W))
			s.V[k] = Zeros(len(p.W))
			mK = s.M[k]
			vK = s.V[k]
			if !mExists || !vExists { s.paramKeys[k] = true }
		}

		for i := range p.W {
			grad := p.Dw[i]
			if math.IsNaN(grad) || math.IsInf(grad, 0) { grad = 0.0; p.Dw[i] = 0.0 }
			mK[i] = s.Beta1*mK[i] + (1.0-s.Beta1)*grad
			vK[i] = s.Beta2*vK[i] + (1.0-s.Beta2)*(grad*grad)
			if math.IsNaN(mK[i]) || math.IsInf(mK[i], 0) { mK[i] = 0 }
			if math.IsNaN(vK[i]) || math.IsInf(vK[i], 0) { vK[i] = 0 }
			denom := math.Sqrt(vK[i]) + s.Eps
			if denom == 0 { continue }
			update := lrT * mK[i] / denom
			if math.IsNaN(update) || math.IsInf(update, 0) { continue }
			p.W[i] -= update
			p.W[i] -= s.LR * s.WD * p.W[i]
		}
		p.ZeroGrads()
	}
}

// GetState extracts the serializable state of the optimizer.
// **No longer needs to return a separate struct for gob, can return SolverAdamW itself**
// **However, keeping the SerializableSolverState struct for clarity and explicit saving**
func (s *SolverAdamW) GetState() SerializableSolverState {
	for key := range s.paramKeys {
		if _, exists := s.M[key]; !exists {
			log.Printf("Warning: Optimizer GetState detected missing M for key %s, initializing.", key)
			s.M[key] = Zeros(0)
		}
		if _, exists := s.V[key]; !exists {
			log.Printf("Warning: Optimizer GetState detected missing V for key %s, initializing.", key)
			s.V[key] = Zeros(0)
		}
	}
	return SerializableSolverState{
		LR:    s.LR,
		Beta1: s.Beta1,
		Beta2: s.Beta2,
		Eps:   s.Eps,
		WD:    s.WD,
		T:     s.T,
		M:     s.M, // M and V are already maps[string][]float64
		V:     s.V,
	}
}

// LoadState configures the optimizer from a saved state.
func (s *SolverAdamW) LoadState(state SerializableSolverState) {
	s.LR = state.LR
	s.Beta1 = state.Beta1
	s.Beta2 = state.Beta2
	s.Eps = state.Eps
	s.WD = state.WD
	s.T = state.T
	s.M = state.M // Assign the loaded maps
	s.V = state.V
	// Rebuild paramKeys from the loaded M map
	s.paramKeys = make(map[string]bool)
	for k := range s.M {
		s.paramKeys[k] = true
	}
	log.Printf("Optimizer state loaded. T=%d, LR=%.e, Beta1=%.3f, Beta2=%.3f, Eps=%.e, WD=%.e, Keys=%d",
		s.T, s.LR, s.Beta1, s.Beta2, s.Eps, s.WD, len(s.paramKeys))
}

//======================================================================
// --- Helper: Create One-Hot Batch Matrix --- (Keep as is or remove if unused)
//======================================================================
func createOneHotBatch(tokenIDs []int, vocabSize int) *Mat {
	currentBatchSize := len(tokenIDs)
	assert(currentBatchSize > 0, "createOneHotBatch requires at least one token ID")
	batchVec := NewMat(vocabSize, currentBatchSize)
	for j, tokenID := range tokenIDs {
		if tokenID >= 0 && tokenID < vocabSize {
			batchVec.Set(tokenID, j, 1.0)
		} else if tokenID != -1 {
			log.Printf("Warning: Index %d out of bounds for one-hot vector size %d in batch item %d.", tokenID, vocabSize, j)
		}
	}
	return batchVec
}

//======================================================================
// --- BPE Training Function ---
//======================================================================
// trainBPEFromFile remains the same
func trainBPEFromFile(bpeDataPath string) error {
	if bpeDataPath == "" { return errors.New("BPE data path is empty") }
	if bpe == nil { return errors.New("global BPE instance is nil") }

	log.Printf("Status: Training BPE tokenizer from '%s'...", bpeDataPath)
	log.Println("\n--- Training BPE ---")

	dataBytes, err := ioutil.ReadFile(bpeDataPath)
	if err != nil {
		log.Printf("Status: Error: Failed to read BPE data file '%s'", bpeDataPath)
		return fmt.Errorf("failed to read BPE data file '%s': %w", bpeDataPath, err)
	}
	bpeCorpus := string(dataBytes)
	if len(strings.TrimSpace(bpeCorpus)) == 0 {
		return fmt.Errorf("BPE data file '%s' is empty or contains only whitespace", bpeDataPath)
	}
	log.Printf("Successfully loaded %d bytes of BPE training data from %s", len(bpeCorpus), bpeDataPath)

	bpeLogWrapper := func(msg string) { log.Println("BPE:", msg) }
	bpe.Train(bpeCorpus, flagBPEVocabSize, false, bpeLogWrapper)
	bpeActualVocabSize = len(bpe.vocabArray)
	if bpeActualVocabSize == 0 { return errors.New("BPE vocab size is zero after training") }
	log.Printf("BPE Actual Vocab Size after training: %d", bpeActualVocabSize)
	log.Println("Status: BPE training complete.")
	return nil
}

//======================================================================
// --- Model Data Preparation (Batching and Shuffling) ---
//======================================================================
// prepareModelData remains the same
func prepareModelData(modelDataPath string) (bool, error) {
	log.Printf("Status: Preparing model training data from file '%s'...", modelDataPath)
	log.Println("\n--- Preparing Model Data ---")
	batches = [][]TrainingSample{}

	if bpe == nil || len(bpe.vocabArray) == 0 {
		return false, errors.New("BPE tokenizer is not initialized or trained before preparing model data")
	}
	log.Printf("Using existing BPE tokenizer with %d vocab size.", bpeActualVocabSize)

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

	encodedTextIDs := bpe.Encode(modelText)
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
		allSamples = append(allSamples, TrainingSample{
			Input:  append([]int{}, inputSeqIDs...),
			Target: append([]int{}, targetSeqIDs...),
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

	rand.Shuffle(len(allSamples), func(i, j int) {
		allSamples[i], allSamples[j] = allSamples[j], allSamples[i]
	})
	log.Println("Shuffled training samples.")

	numBatches := len(allSamples) / currentBatchSize
	batches = make([][]TrainingSample, 0, numBatches)
	for i := 0; i < numBatches; i++ {
		start := i * currentBatchSize
		end := start + currentBatchSize
		batch := allSamples[start:end]
		batches = append(batches, append([]TrainingSample{}, batch...))
	}

	leftoverCount := len(allSamples) % currentBatchSize
	if leftoverCount > 0 {
		log.Printf("Warning: Discarding %d leftover samples that don't form a full batch.", leftoverCount)
	}

	log.Printf("Created %d batches of size %d.", len(batches), currentBatchSize)
	if len(batches) == 0 {
		return false, errors.New("no batches created")
	}

	log.Println("Status: Model data preparation complete.")
	return true, nil
}

//======================================================================
// --- Validation Data Preparation ---
//======================================================================
func prepareValidationData(validationDataPath string) (bool, error) {
	log.Printf("Status: Preparing validation data from file '%s'...", validationDataPath)
	log.Println("\n--- Preparing Validation Data ---")
	validationBatches = [][]TrainingSample{} // Clear previous validation batches

	if bpe == nil || len(bpe.vocabArray) == 0 {
		return false, errors.New("BPE tokenizer is not initialized or trained before preparing validation data")
	}
	log.Printf("Using existing BPE tokenizer with %d vocab size for validation.", bpeActualVocabSize)

	dataBytes, err := ioutil.ReadFile(validationDataPath)
	if err != nil {
		log.Printf("Status: Error: Failed to read validation data file '%s'", validationDataPath)
		return false, fmt.Errorf("failed to read validation data file '%s': %w", validationDataPath, err)
	}
	validationText := string(dataBytes)
	if len(strings.TrimSpace(validationText)) == 0 {
		return false, fmt.Errorf("validation data file '%s' is empty or contains only whitespace", validationDataPath)
	}
	log.Printf("Successfully loaded %d bytes of validation data from %s", len(validationText), validationDataPath)

	encodedTextIDs := bpe.Encode(validationText)
	log.Printf("Encoded validation text -> %d tokens.", len(encodedTextIDs))

	if len(encodedTextIDs) <= seqLength {
		log.Printf("Warning: Encoded validation text length (%d) is not greater than sequence length (%d). Cannot create validation samples.", len(encodedTextIDs), seqLength)
		return true, nil // Not a fatal error, just means no validation possible
	}

	allValidationSamples := []TrainingSample{}
	for i := 0; i <= len(encodedTextIDs)-seqLength-1; i++ {
		inputSeqIDs := encodedTextIDs[i : i+seqLength]
		targetSeqIDs := encodedTextIDs[i+1 : i+seqLength+1]
		allValidationSamples = append(allValidationSamples, TrainingSample{
			Input:  append([]int{}, inputSeqIDs...),
			Target: append([]int{}, targetSeqIDs...),
		})
	}

	log.Println("Total individual validation sequences generated:", len(allValidationSamples))
	if len(allValidationSamples) == 0 {
		log.Println("Warning: No validation sequences generated.")
		return true, nil // Not fatal
	}

	// Use the same batch size as training for simplicity
	currentBatchSize := batchSize
	if len(allValidationSamples) < currentBatchSize {
		log.Printf("Warning: Number of validation samples (%d) is less than configured batch size (%d). Using a smaller batch size for validation.", len(allValidationSamples), currentBatchSize)
		currentBatchSize = len(allValidationSamples)
		if currentBatchSize == 0 {
			return true, nil // No samples, no batches
		}
	}

	// No shuffling for validation data
	numBatches := len(allValidationSamples) / currentBatchSize
	validationBatches = make([][]TrainingSample, 0, numBatches)
	for i := 0; i < numBatches; i++ {
		start := i * currentBatchSize
		end := start + currentBatchSize
		batch := allValidationSamples[start:end]
		validationBatches = append(validationBatches, append([]TrainingSample{}, batch...))
	}

	leftoverCount := len(allValidationSamples) % currentBatchSize
	if leftoverCount > 0 {
		log.Printf("Info: Discarding %d leftover validation samples that don't form a full batch.", leftoverCount)
		// Alternatively, could create one smaller final batch
	}

	log.Printf("Created %d validation batches of size up to %d.", len(validationBatches), currentBatchSize)
	if len(validationBatches) == 0 {
		log.Println("Warning: No validation batches created.")
	}

	log.Println("Status: Validation data preparation complete.")
	return true, nil
}


//======================================================================
// --- Checkpointing Structures and Functions ---
//======================================================================
// SerializableMat needs exported fields for gob
type SerializableMat struct {
	N  int
	D  int
	W  []float64
	Dw []float64 // Keep Dw for resuming training
}

// SerializableSolverState needs exported fields for gob
type SerializableSolverState struct {
	LR    float64
	Beta1 float64
	Beta2 float64
	Eps   float64
	WD    float64
	T     int
	M     map[string][]float64
	V     map[string][]float64
}

// Checkpoint struct needs exported fields for gob
type Checkpoint struct {
	Epoch          int
	ModelParams    map[string]SerializableMat
	OptimizerState SerializableSolverState
	BPEState       BPESavedState
	Config         struct {
		BPEVocabSize       int
		EmbeddingDimension int
		GRUHiddenSize      int
		GRULayers          int
		NumExperts         int
		TrainSeqLength     int
		BatchSize          int
		Epochs             int
		MaxResponseLength  int
		LearningRate       float64
		WeightDecay        float64
		EpsilonRMSNorm     float64
		EpsilonAdamW       float64
		GradientClipValue  float64
		BPEActualVocabSize int
		HiddenSizes        []int
	}
}

// matToSerializable remains the same
func matToSerializable(m *Mat) SerializableMat {
	wCopy := make([]float64, len(m.W))
	dwCopy := make([]float64, len(m.Dw))
	copy(wCopy, m.W)
	copy(dwCopy, m.Dw)
	return SerializableMat{
		N:  m.N,
		D:  m.D,
		W:  wCopy,
		Dw: dwCopy,
	}
}

// serializableToMat remains the same
func serializableToMat(sm SerializableMat) *Mat {
	m := NewMat(sm.N, sm.D)
	copy(m.W, sm.W)
	if len(sm.Dw) == len(m.Dw) {
		copy(m.Dw, sm.Dw)
	} else if len(sm.Dw) != 0 {
		log.Printf("Warning: Checkpoint Dw size (%d) mismatch for matrix %dx%d (expected %d), gradients not loaded.", len(sm.Dw), sm.N, sm.D, len(m.Dw))
	}
	return m
}

// --- Gob Type Registration ---
// Register the types that will be encoded/decoded in the checkpoint.
// This is crucial for gob, especially for nested complex types like maps and custom structs.
func init() {
	gob.Register(Checkpoint{})
	gob.Register(SerializableMat{})
	gob.Register(SerializableSolverState{})
	gob.Register(BPESavedState{})
	gob.Register(map[string]SerializableMat{})
	gob.Register(map[string][]float64{})
	gob.Register(map[string]MergeInfo{}) // Register MergeInfo as it's part of BPESavedState.Merges map value
	gob.Register(MergeInfo{})            // Also register the struct itself
}

// saveCheckpoint saves the current training state using gob.
func saveCheckpoint(epoch int, model map[string]*Mat, solver *SolverAdamW, bpe *BPE, path string) error {
	log.Printf("Saving checkpoint for epoch %d to %s...", epoch, path)

	// Ensure the directory exists
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create checkpoint directory %s: %w", dir, err)
	}

	// 1. Convert Model Parameters
	serializableModel := make(map[string]SerializableMat)
	for k, v := range model {
		serializableModel[k] = matToSerializable(v)
	}

	// 2. Get Optimizer State
	optimizerState := solver.GetState()

	// 3. Get BPE State
	bpeState := bpe.GetState()

	// 4. Create Checkpoint struct
	checkpoint := Checkpoint{
		Epoch:          epoch,
		ModelParams:    serializableModel,
		OptimizerState: optimizerState,
		BPEState:       bpeState,
	}
	// Add config values from the flag variables
	checkpoint.Config.BPEVocabSize = flagBPEVocabSize
	checkpoint.Config.EmbeddingDimension = flagEmbeddingDimension
	checkpoint.Config.GRUHiddenSize = flagGRUHiddenSize
	checkpoint.Config.GRULayers = flagGRULayers
	checkpoint.Config.NumExperts = flagNumExperts
	checkpoint.Config.TrainSeqLength = flagTrainSeqLength
	checkpoint.Config.BatchSize = flagBatchSize
	checkpoint.Config.Epochs = flagEpochs
	checkpoint.Config.MaxResponseLength = flagMaxResponseLength
	checkpoint.Config.LearningRate = flagLearningRate
	checkpoint.Config.WeightDecay = flagWeightDecay
	checkpoint.Config.EpsilonRMSNorm = flagEpsilonRMSNorm
	checkpoint.Config.EpsilonAdamW = flagEpsilonAdamW
	checkpoint.Config.GradientClipValue = flagGradientClipValue
	checkpoint.Config.BPEActualVocabSize = bpeActualVocabSize
	checkpoint.Config.HiddenSizes = append([]int{}, hiddenSizes...)

	// 5. Write to file atomically using gob
	tempPath := path + ".tmp"
	file, err := os.Create(tempPath)
	if err != nil {
		return fmt.Errorf("failed to create temporary checkpoint file %s: %w", tempPath, err)
	}
	defer file.Close() // Ensure file is closed

	encoder := gob.NewEncoder(file)
	err = encoder.Encode(checkpoint)
	if err != nil {
		// Close the file before trying to remove it
		file.Close()
		_ = os.Remove(tempPath) // Attempt to clean up temp file
		return fmt.Errorf("failed to encode checkpoint data to %s: %w", tempPath, err)
	}

	// Close the file explicitly before renaming
	if err := file.Close(); err != nil {
		_ = os.Remove(tempPath) // Attempt cleanup
		return fmt.Errorf("failed to close temporary checkpoint file %s before rename: %w", tempPath, err)
	}

	// 6. Rename temporary file to final path
	err = os.Rename(tempPath, path)
	if err != nil {
		_ = os.Remove(tempPath) // Attempt cleanup if rename fails
		return fmt.Errorf("failed to rename temporary checkpoint file to %s: %w", path, err)
	}

	log.Printf("Checkpoint saved successfully to %s", path)
	return nil
}

// loadCheckpoint loads training state using gob and updates global config vars.
func loadCheckpoint(path string) (startEpoch int, loadedModel map[string]*Mat, loadedSolver *SolverAdamW, loadedBPE *BPE, err error) {
	log.Printf("Loading checkpoint from %s...", path)

	// 1. Open file
	file, err := os.Open(path)
	if err != nil {
		err = fmt.Errorf("failed to open checkpoint file %s: %w", path, err)
		return
	}
	defer file.Close()

	// 2. Decode gob data
	var checkpoint Checkpoint
	decoder := gob.NewDecoder(file)
	err = decoder.Decode(&checkpoint)
	if err != nil {
		err = fmt.Errorf("failed to decode gob checkpoint data from %s: %w", path, err)
		return
	}

	// 3. Validate Config against current flag values (informational) - Same as before
	log.Println("Validating checkpoint configuration against current flag settings...")
	configMismatch := false
	if checkpoint.Config.EmbeddingDimension != flagEmbeddingDimension {
		log.Printf("Warning: Checkpoint EmbeddingDimension (%d) differs from current flag (%d)", checkpoint.Config.EmbeddingDimension, flagEmbeddingDimension); configMismatch=true
	}
	if len(checkpoint.Config.HiddenSizes) != flagGRULayers {
		log.Printf("Warning: Checkpoint GRULayers based on HiddenSizes length (%d) differs from current flag (%d)", len(checkpoint.Config.HiddenSizes), flagGRULayers); configMismatch=true
	} else {
		currentHiddenSizes := make([]int, flagGRULayers)
		for i := range currentHiddenSizes { currentHiddenSizes[i] = flagGRUHiddenSize }
		for i := range currentHiddenSizes {
			if checkpoint.Config.HiddenSizes[i] != currentHiddenSizes[i] {
				log.Printf("Warning: Checkpoint HiddenSize[%d] (%d) differs from current flag (%d)", i, checkpoint.Config.HiddenSizes[i], currentHiddenSizes[i]); configMismatch=true; break
			}
		}
	}
	if checkpoint.Config.NumExperts != flagNumExperts {
		log.Printf("Warning: Checkpoint NumExperts (%d) differs from current flag (%d)", checkpoint.Config.NumExperts, flagNumExperts); configMismatch=true
	}
	if checkpoint.Config.TrainSeqLength != flagTrainSeqLength {
		log.Printf("Warning: Checkpoint TrainSeqLength (%d) differs from current flag (%d)", checkpoint.Config.TrainSeqLength, flagTrainSeqLength); configMismatch=true
	}
	log.Printf("Info: Checkpoint BPE Actual Vocab Size: %d", checkpoint.Config.BPEActualVocabSize)
	if math.Abs(checkpoint.Config.LearningRate-flagLearningRate) > 1e-9 {
		log.Printf("Warning: Checkpoint LearningRate (%.e) differs from current flag (%.e)", checkpoint.Config.LearningRate, flagLearningRate); configMismatch=true
	}
	if math.Abs(checkpoint.Config.WeightDecay-flagWeightDecay) > 1e-9 {
		log.Printf("Warning: Checkpoint WeightDecay (%.e) differs from current flag (%.e)", checkpoint.Config.WeightDecay, flagWeightDecay); configMismatch=true
	}
	if configMismatch {
		log.Println("Configuration mismatch detected. Checkpoint values will be used for model structure and optimizer state. Current flags may affect subsequent training or behavior if applicable.")
	} else {
		log.Println("Checkpoint configuration broadly matches current flag settings.")
	}

	// 4. Reconstruct Model - Same as before
	loadedModel = make(map[string]*Mat)
	for k, sm := range checkpoint.ModelParams {
		loadedModel[k] = serializableToMat(sm)
	}
	log.Printf("Loaded %d model parameters.", len(loadedModel))

	// 5. Reconstruct Optimizer - Same as before
	loadedSolver = NewSolverAdamW(
		checkpoint.OptimizerState.LR,
		checkpoint.OptimizerState.Beta1,
		checkpoint.OptimizerState.Beta2,
		checkpoint.OptimizerState.Eps,
		checkpoint.OptimizerState.WD,
	)
	loadedSolver.LoadState(checkpoint.OptimizerState)

	// 6. Reconstruct BPE Tokenizer - Same as before
	loadedBPE = NewBPE(checkpoint.BPEState.SpecialTokens)
	err = loadedBPE.LoadState(checkpoint.BPEState)
	if err != nil {
		err = fmt.Errorf("failed to load BPE state from checkpoint: %w", err)
		return
	}
	bpeActualVocabSize = len(loadedBPE.vocabArray)
	log.Printf("Loaded BPE tokenizer with %d vocab size.", bpeActualVocabSize)

	// 7. Update Global Configuration Variables from Checkpoint Config - Same as before
	log.Println("Applying checkpoint configuration to runtime variables...")
	embeddingDimension = checkpoint.Config.EmbeddingDimension
	hiddenSizes = append([]int{}, checkpoint.Config.HiddenSizes...)
	numExperts = checkpoint.Config.NumExperts
	seqLength = checkpoint.Config.TrainSeqLength
	batchSize = checkpoint.Config.BatchSize
	flagMaxResponseLength = checkpoint.Config.MaxResponseLength
	flagGRULayers = len(hiddenSizes)
	if len(hiddenSizes) > 0 { flagGRUHiddenSize = hiddenSizes[0] }
	flagLearningRate = loadedSolver.LR
	flagWeightDecay = loadedSolver.WD
	flagEpsilonAdamW = loadedSolver.Eps
	flagEpsilonRMSNorm = checkpoint.Config.EpsilonRMSNorm
	flagGradientClipValue = checkpoint.Config.GradientClipValue
	flagEpochs = checkpoint.Config.Epochs

	// 8. Return loaded state - Same as before
	startEpoch = checkpoint.Epoch + 1
	log.Printf("Checkpoint loaded successfully. Configuration updated. Resuming from epoch %d.", startEpoch)
	return // Returns named return values (startEpoch, loadedModel, loadedSolver, loadedBPE, err=nil)
}


//======================================================================
// --- Validation Loss Calculation ---
//======================================================================
func calculateValidationLoss(model map[string]*Mat, valBatches [][]TrainingSample) (float64, error) {
	if len(valBatches) == 0 {
		log.Println("Info: No validation batches available to calculate loss.")
		return 0.0, nil // No error, just no data
	}
	if model == nil {
		return 0.0, errors.New("validation loss calculation called but model is nil")
	}
	if bpeActualVocabSize <= 0 {
		return 0.0, errors.New("validation loss calculation called but BPE vocab size is zero")
	}

	log.Printf("Status: Calculating validation loss on %d batches...", len(valBatches))
	startTime := time.Now()

	totalValidationLoss := 0.0
	totalValidValidationSteps := 0

	for batchIndex, batch := range valBatches {
		currentBatchSize := len(batch)
		if currentBatchSize == 0 { continue }

		// Use a graph that DOES NOT track gradients
		g := NewGraph(false)
		var hiddenStates [][]*Mat // Reset for each batch
		batchLoss := 0.0
		validStepsInBatch := 0

		for t := 0; t < seqLength; t++ {
			inputTokenIDs := make([]int, currentBatchSize)
			targetTokenIDs := make([]int, currentBatchSize)
			hasValidTargetInStep := false

			for i := 0; i < currentBatchSize; i++ {
				if t < len(batch[i].Input) && t < len(batch[i].Target) {
					if batch[i].Input[t] >= 0 && batch[i].Input[t] < bpeActualVocabSize {
						inputTokenIDs[i] = batch[i].Input[t]
					} else { inputTokenIDs[i] = -1 } // Use -1 for lookup handling
					if batch[i].Target[t] >= 0 && batch[i].Target[t] < bpeActualVocabSize {
						targetTokenIDs[i] = batch[i].Target[t]
						hasValidTargetInStep = true
					} else { targetTokenIDs[i] = -1 }
				} else { inputTokenIDs[i] = -1; targetTokenIDs[i] = -1 }
			}

			if !hasValidTargetInStep { continue } // Skip step if no valid targets in batch

			// Forward pass only
			xBatch := g.Lookup(model["WE"], inputTokenIDs) // Handles -1 indices internally
			forwardResult := ForwardMoEGRU(g, model, hiddenSizes, numExperts, xBatch, hiddenStates)
			hiddenStates = forwardResult.H // Update hidden states for next step
			outputLogits := forwardResult.O

			// Calculate probabilities (no backprop needed)
			probs := SoftmaxStandalone(outputLogits)

			stepLoss := 0.0
			numValidInStep := 0

			for j := 0; j < currentBatchSize; j++ {
				targetTokenID := targetTokenIDs[j]
				if targetTokenID == -1 { continue } // Skip if target invalid

				targetProb := probs.Get(targetTokenID, j)
				loss := -math.Log(math.Max(targetProb, 1e-9)) // Use Max for stability

				if math.IsNaN(loss) || math.IsInf(loss, 0) {
					log.Printf("Warn: NaN/Inf validation loss in Batch %d, Item %d, Step %d. TargetID: %d, Prob: %.4e. Skipping item's contribution.", batchIndex, j, t, targetTokenID, targetProb)
					continue // Skip this item's loss contribution
				}

				numValidInStep++
				stepLoss += loss
			} // End loop over batch items (j)

			if numValidInStep > 0 {
				batchLoss += stepLoss
				validStepsInBatch += numValidInStep
				// No gradient accumulation or scaling needed
			}
		} // End sequence loop (t)

		if validStepsInBatch > 0 && !math.IsNaN(batchLoss) && !math.IsInf(batchLoss, 0) {
			totalValidationLoss += batchLoss
			totalValidValidationSteps += validStepsInBatch
		} else if validStepsInBatch > 0 {
			log.Printf("Warn: Invalid total validation batch loss (%.4f) despite %d valid steps in Batch %d.", batchLoss, validStepsInBatch, batchIndex)
		}

	} // End batch loop (batchIndex)

	avgValidationLoss := 0.0
	if totalValidValidationSteps > 0 {
		avgValidationLoss = totalValidationLoss / float64(totalValidValidationSteps)
	} else {
		log.Println("Warning: Validation completed with zero valid steps across all batches.")
		return 0.0, nil // Return 0 loss if no valid steps
	}

	duration := time.Since(startTime)
	log.Printf("Status: Validation loss calculation complete. Avg Loss: %.4f, Duration: %s", avgValidationLoss, duration)

	return avgValidationLoss, nil
}


//======================================================================
// --- Training Loop ---
//======================================================================
func trainGRUModel(startEpoch int) error {
	if model == nil || solver == nil || bpe == nil {
		return errors.New("training called but model, solver, or BPE is not initialized")
	}
	if bpeActualVocabSize <= 0 {
		return errors.New("training called but BPE vocab size is zero")
	}

	log.Printf("Status: starting from epoch %d...", startEpoch)
	log.Println("\n--- Training model ---")

	totalBatches := len(batches)
	if totalBatches == 0 { return errors.New("no batches found for training") }
	log.Printf("Starting training: %d total epochs configured, %d batches/epoch, Batch Size: %d, Embedding Dim: %d...", flagEpochs, totalBatches, batchSize, embeddingDimension)

	for epoch := startEpoch; epoch < flagEpochs; epoch++ {
		log.Printf("\nStatus: Starting Epoch %d/%d", epoch+1, flagEpochs) // Add newline for better separation
		epochStartTime := time.Now()
		cumulativeEpochLoss := 0.0
		totalValidStepsInEpoch := 0

		rand.Shuffle(len(batches), func(i, j int) { batches[i], batches[j] = batches[j], batches[i] })
		progressInterval := totalBatches / 20
		if progressInterval == 0 { progressInterval = 1 }

		// --- Training Batch Loop ---
		for batchIndex, batch := range batches {
			currentBatchSize := len(batch)
			if currentBatchSize == 0 { continue }

			g := NewGraph(true)
			var hiddenStates [][]*Mat // Reset for each batch
			batchLoss := 0.0
			validStepsInBatch := 0

			for t := 0; t < seqLength; t++ {
				inputTokenIDs := make([]int, currentBatchSize)
				targetTokenIDs := make([]int, currentBatchSize)
				hasValidTargetInStep := false

				for i := 0; i < currentBatchSize; i++ {
					if t < len(batch[i].Input) && t < len(batch[i].Target) {
						if batch[i].Input[t] >= 0 && batch[i].Input[t] < bpeActualVocabSize {
							inputTokenIDs[i] = batch[i].Input[t]
						} else { inputTokenIDs[i] = -1 } // Use -1 for lookup handling
						if batch[i].Target[t] >= 0 && batch[i].Target[t] < bpeActualVocabSize {
							targetTokenIDs[i] = batch[i].Target[t]
							hasValidTargetInStep = true
						} else { targetTokenIDs[i] = -1 }
					} else { inputTokenIDs[i] = -1; targetTokenIDs[i] = -1 }
				}

				if !hasValidTargetInStep { continue } // Skip step if no valid targets

				xBatch := g.Lookup(model["WE"], inputTokenIDs) // Handles -1 indices
				forwardResult := ForwardMoEGRU(g, model, hiddenSizes, numExperts, xBatch, hiddenStates)
				hiddenStates = forwardResult.H
				outputLogits := forwardResult.O
				probs := SoftmaxStandalone(outputLogits) // Use standalone for loss calc

				stepLoss := 0.0
				dLdLogits := NewMat(bpeActualVocabSize, currentBatchSize) // Gradient w.r.t logits
				numValidInStep := 0

				for j := 0; j < currentBatchSize; j++ {
					targetTokenID := targetTokenIDs[j]
					if targetTokenID == -1 { continue } // Skip if target invalid

					targetProb := probs.Get(targetTokenID, j)
					loss := -math.Log(math.Max(targetProb, 1e-9)) // Use Max for stability

					if math.IsNaN(loss) || math.IsInf(loss, 0) {
						log.Printf("Warn: NaN/Inf training loss Ep %d, Batch %d, Item %d, Step %d. TargetID: %d, Prob: %.4e. Skipping item.", epoch+1, batchIndex, j, t, targetTokenID, targetProb)
						continue // Skip this item's loss and gradient contribution
					}

					numValidInStep++
					stepLoss += loss

					// Calculate gradient for backprop (dL/dLogits = Probs - Target)
					for i := 0; i < bpeActualVocabSize; i++ {
						delta := probs.Get(i, j)
						if i == targetTokenID { delta -= 1.0 }
						if !math.IsNaN(delta) && !math.IsInf(delta, 0) {
							dLdLogits.Set(i, j, delta)
						} else {
							dLdLogits.Set(i, j, 0.0) // Zero out invalid gradients
						}
					}
				} // End loop over batch items (j)

				if numValidInStep > 0 {
					batchLoss += stepLoss
					validStepsInBatch += numValidInStep

					// Scale gradients by 1/numValidInStep and apply to outputLogits.Dw
					scaleFactor := 1.0 / float64(numValidInStep)
					for j := 0; j < currentBatchSize; j++ {
						if targetTokenIDs[j] != -1 { // Only apply gradient if target was valid
							for i := 0; i < bpeActualVocabSize; i++ {
								grad_ij := dLdLogits.Get(i, j)
								// Check grad_ij again before adding to Dw
								if !math.IsNaN(grad_ij) && !math.IsInf(grad_ij, 0) {
									outputLogits.Dw[i*currentBatchSize+j] += grad_ij * scaleFactor
								}
							}
						}
					}
				}
			} // End sequence loop (t)

			if validStepsInBatch > 0 && !math.IsNaN(batchLoss) && !math.IsInf(batchLoss, 0) {
				g.Backward() // Perform backpropagation

				// Gradient Clipping
				params := GetModelParameters(model)
				var gradNormSq float64 = 0
				for _, p := range params {
					for _, dwVal := range p.Dw {
						if !math.IsNaN(dwVal) && !math.IsInf(dwVal, 0) { gradNormSq += dwVal * dwVal }
					}
				}

				if !math.IsNaN(gradNormSq) && !math.IsInf(gradNormSq, 0) && gradNormSq > 0 {
					gradNorm := math.Sqrt(gradNormSq)
					if gradNorm > flagGradientClipValue {
						scale := flagGradientClipValue / (gradNorm + 1e-7) // Add epsilon for safety
						for _, p := range params {
							for i := range p.Dw {
								if !math.IsNaN(p.Dw[i]) && !math.IsInf(p.Dw[i], 0) {
									p.Dw[i] *= scale
								} else {
									p.Dw[i] = 0 // Zero out invalid gradients after scaling attempt
								}
							}
						}
					}
					// Optimizer Step
					solver.Step(model) // Updates weights and zeros grads
				} else {
					log.Printf("Warn: Grad norm invalid (sqrt(%.4f)) or zero Ep %d Batch %d. Zeroing grads and skipping optimizer step.", gradNormSq, epoch+1, batchIndex)
					ZeroModelGrads(model) // Zero grads manually if optimizer step skipped
				}

				cumulativeEpochLoss += batchLoss
				totalValidStepsInEpoch += validStepsInBatch

			} else if validStepsInBatch > 0 {
				log.Printf("Warn: Invalid batch loss (%.4f) despite %d valid steps Ep %d Batch %d. Zeroing grads.", batchLoss, validStepsInBatch, epoch+1, batchIndex)
				ZeroModelGrads(model) // Zero grads if loss was invalid
			} else {
				// No valid steps in batch, no loss, no grads to zero or step.
			}

			// Progress Bar Update
			if (batchIndex+1)%progressInterval == 0 || batchIndex == totalBatches-1 {
				doneCount := batchIndex + 1
				percentage := float64(doneCount) / float64(totalBatches) * 100
				barLength := 20
				filledLength := int(percentage / 100 * float64(barLength))
				if filledLength > barLength { filledLength = barLength }
				if filledLength < 0 { filledLength = 0 }
				bar := strings.Repeat("=", filledLength) + strings.Repeat("-", barLength-filledLength)
				
				currentAvgLoss := 0.0
				if totalValidStepsInEpoch > 0 {
					currentAvgLoss = cumulativeEpochLoss / float64(totalValidStepsInEpoch)
				}
				
				fmt.Printf("\rEpoch %d/%d [%s] %d/%d (%.1f%%) Avg Loss: %.4f", epoch+1, flagEpochs, bar, doneCount, totalBatches, percentage, currentAvgLoss)
			}

		} // End batch loop (batchIndex)
		fmt.Println() // Final newline after progress bar

		avgEpochLoss := 0.0
		if totalValidStepsInEpoch > 0 {
			avgEpochLoss = cumulativeEpochLoss / float64(totalValidStepsInEpoch)
		} else {
			log.Printf("Warning: Epoch %d completed with zero valid training steps.", epoch+1)
		}
		epochDuration := time.Since(epochStartTime)
		log.Printf("Epoch: %d/%d, Average Training Step Loss: %.4f, Duration: %s", epoch+1, flagEpochs, avgEpochLoss, epochDuration)

		// --- Validation Step ---
		if len(validationBatches) > 0 {
			validationLoss, valErr := calculateValidationLoss(model, validationBatches)
			if valErr != nil {
				log.Printf("Error calculating validation loss for epoch %d: %v", epoch+1, valErr)
				// Continue training even if validation fails for an epoch
			} else {
				log.Printf("Epoch: %d/%d, Validation Loss: %.4f", epoch+1, flagEpochs, validationLoss)
			}
		} else {
			log.Printf("Epoch: %d/%d, No validation data provided.", epoch+1, flagEpochs)
		}
		// --- End Validation Step ---


		// --- Save Checkpoint ---
		checkpointFilename := fmt.Sprintf("checkpoint_epoch_%d.gob", epoch)
		checkpointFilepath := filepath.Join(CheckpointDir, checkpointFilename)
		err := saveCheckpoint(epoch, model, solver, bpe, checkpointFilepath)
		if err != nil {
			log.Printf("Error saving checkpoint for epoch %d: %v", epoch, err)
			// Continue training for now
		}
	} // End Epoch Loop

	log.Println("--- Training Complete ---")
	log.Println("Status: Training finished. Ready for chat.")
	trainingComplete = true
	return nil
}


//======================================================================
// --- Conversational Response Generation ---
//======================================================================
// generateResponse remains the same logically
func generateResponse(inputText string, maxLength int) (string, error) {
	if !trainingComplete || bpe == nil || model == nil {
		return "Sorry, the model hasn't been trained or loaded yet.", nil
	}
	if numExperts <= 0 { return "Error: Model configuration issue (numExperts invalid).", errors.New("numExperts invalid") }
	if _, ok := model["WE"]; !ok { return "Error: Model configuration issue (WE embedding missing).", errors.New("WE missing") }
	if bpeActualVocabSize <= 0 { return "Error: BPE tokenizer not properly initialized (vocab size 0).", errors.New("BPE vocab size 0") }

	g := NewGraph(false)
	var hiddenStates [][]*Mat

	userToken := "[USER]"; botToken := "[BOT]"; eosToken := "[EOS]"
	userTokenID, hasUser := bpe.specialTokensMap[userToken]
	botTokenID, hasBot := bpe.specialTokensMap[botToken]
	eosTokenID, hasEOS := bpe.specialTokensMap[eosToken]
	unkTokenID, hasUnk := bpe.specialTokensMap["[UNK]"]

	promptText := fmt.Sprintf("%s %s %s", userToken, inputText, botToken)
	promptIDs := bpe.Encode(promptText)
	validPromptIDsForPriming := []int{}
	for _, id := range promptIDs {
		if id >= 0 && id < bpeActualVocabSize {
			validPromptIDsForPriming = append(validPromptIDsForPriming, id)
		} else {
			log.Printf("Warning: Invalid token ID %d in prompt, treating as UNK/skipping in priming.", id)
			if hasUnk { validPromptIDsForPriming = append(validPromptIDsForPriming, unkTokenID) } else { validPromptIDsForPriming = append(validPromptIDsForPriming, -1) } // Use -1 if no UNK
		}
	}
	if len(validPromptIDsForPriming) == 0 {
		log.Println("Warning: No valid tokens found after encoding the prompt.")
		return "I couldn't process that input.", nil
	}

	currentTokenID := -1
	// Prime the model with the prompt
	for _, tokenID := range validPromptIDsForPriming {
		if tokenID == -1 { continue } // Skip invalid tokens during priming forward pass
		x := g.Lookup(model["WE"], []int{tokenID})
		forwardResult := ForwardMoEGRU(g, model, hiddenSizes, numExperts, x, hiddenStates)
		hiddenStates = forwardResult.H
		currentTokenID = tokenID // Update currentTokenID with the last valid token processed
	}
	if currentTokenID == -1 {
		log.Println("Error: Failed to set currentTokenId during priming (only invalid tokens?). Using BOT token as fallback.")
		// Fallback: If priming failed entirely, start generation from BOT token
		if hasBot {
			currentTokenID = botTokenID
		} else {
			return "Error processing the input prompt (priming failed).", errors.New("failed to set currentTokenId during priming, no BOT fallback")
		}
	}


	generatedResponseIDs := []int{}
	for t := 0; t < maxLength; t++ {
		if currentTokenID < 0 || currentTokenID >= bpeActualVocabSize {
			log.Printf("Error: Invalid currentTokenId (%d) at start of generation step %d. Stopping.", currentTokenID, t)
			break
		}
		x := g.Lookup(model["WE"], []int{currentTokenID})
		forwardResult := ForwardMoEGRU(g, model, hiddenSizes, numExperts, x, hiddenStates)
		hiddenStates = forwardResult.H // Update hidden state for next step
		outputLogits := forwardResult.O
		probs := SoftmaxStandalone(outputLogits) // Get probabilities for the next token

		// Sampling logic
		sample := rand.Float64()
		cumulativeProb := 0.0
		nextTokenID := -1

		// Check and potentially renormalize probabilities
		probSum := 0.0; validProbs := true
		for i := 0; i < probs.N; i++ {
			probVal := probs.Get(i, 0)
			if math.IsNaN(probVal) || math.IsInf(probVal, 0) {
				probs.Set(i, 0, 0.0); validProbs = false // Zero out invalid probs
			}
			probSum += probs.Get(i, 0)
		}

		if !validProbs || math.Abs(probSum-1.0) > 1e-5 { // If probs were invalid or sum is off
			log.Printf("Warning: Probabilities invalid or sum %.5f in step %d. Renormalizing/Uniform Sampling.", probSum, t)
			if probSum <= 1e-9 { // If sum is effectively zero, sample uniformly
				nextTokenID = rand.Intn(bpeActualVocabSize)
				goto EndSampling // Skip standard sampling loop
			}
			// Renormalize
			renormFactor := 1.0 / probSum
			cumulativeProb = 0.0
			for i := 0; i < probs.N; i++ {
				renormalizedProb := probs.Get(i, 0) * renormFactor
				probs.Set(i, 0, renormalizedProb) // Update matrix in place (though not strictly necessary for sampling)
				cumulativeProb += renormalizedProb
				if sample < cumulativeProb && nextTokenID == -1 { // Find first token crossing threshold
					nextTokenID = i
				}
			}
			if nextTokenID == -1 { // Should not happen if probSum > 0, but as safeguard
				nextTokenID = bpeActualVocabSize - 1
			}
			// goto EndSampling // Already handled by loop break/assignment
		} else {
			// Standard sampling from valid probability distribution
			for i := 0; i < bpeActualVocabSize; i++ {
				cumulativeProb += probs.Get(i, 0)
				if sample < cumulativeProb {
					nextTokenID = i
					break
				}
			}
			if nextTokenID == -1 { // If sampling failed somehow (e.g., rounding errors), pick last token
				nextTokenID = bpeActualVocabSize - 1
			}
		}

	EndSampling: // Label for goto jump if needed (e.g., uniform sampling)
		// Stop generation if EOS or USER token is sampled
		if (hasEOS && nextTokenID == eosTokenID) || (hasUser && nextTokenID == userTokenID) {
			break
		}
		if nextTokenID < 0 || nextTokenID >= bpeActualVocabSize {
			log.Printf("Error: Sampled invalid token ID %d in step %d. Stopping generation.", nextTokenID, t)
			break
		}

		generatedResponseIDs = append(generatedResponseIDs, nextTokenID)
		currentTokenID = nextTokenID // Set the sampled token as the input for the next step
	}

	if len(generatedResponseIDs) == 0 { return "...", nil } // Return placeholder if nothing generated

	decodedString := bpe.Decode(generatedResponseIDs)

	// Optional: Clean up potential leading BOT token if generation started with it
	if hasBot && len(generatedResponseIDs) > 0 && generatedResponseIDs[0] == botTokenID {
		botTokenString := ""
		if botTokenID >= 0 && botTokenID < len(bpe.vocabArray) {
			botTokenString = bpe.vocabArray[botTokenID]
			// Be careful with decoding; sometimes spaces are handled differently.
			// Check prefix based on the raw token string.
			if strings.HasPrefix(decodedString, botTokenString) {
				decodedString = strings.TrimPrefix(decodedString, botTokenString)
				decodedString = strings.TrimSpace(decodedString) // Trim potential space after token
			}
		}
	}


	finalResponse := strings.TrimSpace(decodedString)
	if finalResponse == "" { finalResponse = "..." } // Final check for empty response
	return finalResponse, nil
}


//======================================================================
// --- Main Execution & Chat Interface ---
//======================================================================

func main() {
	rand.Seed(time.Now().UnixNano())
	log.SetFlags(log.LstdFlags | log.Lshortfile) // Add file/line number to logs

	// --- Define Flags ---
	flag.IntVar(&flagBPEVocabSize, "bpe-vocab-size", 850, "Target vocabulary size for BPE training")
	flag.IntVar(&flagEmbeddingDimension, "embedding-dim", 96, "Dimension for token embeddings")
	flag.IntVar(&flagGRUHiddenSize, "gru-hidden-size", 96, "Hidden size for GRU layers (used if gru-layers > 0)")
	flag.IntVar(&flagGRULayers, "gru-layers", 2, "Number of GRU layers")
	flag.IntVar(&flagNumExperts, "num-experts", 6, "Number of experts in MoE layers")
	flag.IntVar(&flagTrainSeqLength, "seq-length", 80, "Sequence length for training")
	flag.IntVar(&flagBatchSize, "batch-size", 16, "Batch size for training")
	flag.IntVar(&flagEpochs, "epochs", 5, "Number of training epochs")
	flag.IntVar(&flagMaxResponseLength, "max-response", 260, "Maximum number of tokens to generate in response")
	flag.Float64Var(&flagLearningRate, "lr", 0.001, "Learning rate for AdamW optimizer")
	flag.Float64Var(&flagWeightDecay, "wd", 0.01, "Weight decay for AdamW optimizer")
	flag.Float64Var(&flagEpsilonRMSNorm, "eps-rmsnorm", 1e-5, "Epsilon for RMSNorm stability")
	flag.Float64Var(&flagEpsilonAdamW, "eps-adamw", 1e-8, "Epsilon for AdamW optimizer stability")
	flag.Float64Var(&flagGradientClipValue, "grad-clip", 5.0, "Gradient clipping value")
	checkpointFlag := flag.String("checkpoint", "", "Path to checkpoint file (.gob) to load and resume training/inference")
	bpeDataFlag := flag.String("bpe-data", "", "Path to the data file for BPE tokenizer training")
	modelDataFlag := flag.String("model-data", "", "Path to the data file for model training")
	validationDataFlag := flag.String("validation-data", "", "Path to the data file for validation loss calculation after each epoch") // <-- ADDED
	trainFlag := flag.Bool("train", false, "Run/continue model training (requires -model-data)")

	// --- Parse Flags ---
	flag.Parse()

	// --- Post-Flag Setup & Variable Assignment ---
	embeddingDimension = flagEmbeddingDimension
	hiddenSizes = make([]int, flagGRULayers)
	for i := range hiddenSizes { hiddenSizes[i] = flagGRUHiddenSize }
	numExperts = flagNumExperts
	seqLength = flagTrainSeqLength
	batchSize = flagBatchSize

	log.Println("--- Effective Configuration (Initial / Flags) ---")
	log.Printf("  BPEVocabSize: %d", flagBPEVocabSize)
	log.Printf("  EmbeddingDimension: %d", embeddingDimension)
	log.Printf("  GRUHiddenSize: %d", flagGRUHiddenSize)
	log.Printf("  GRULayers: %d", flagGRULayers)
	log.Printf("  NumExperts: %d", numExperts)
	log.Printf("  TrainSeqLength: %d", seqLength)
	log.Printf("  BatchSize: %d", batchSize)
	log.Printf("  Epochs: %d", flagEpochs)
	log.Printf("  MaxResponseLength: %d", flagMaxResponseLength)
	log.Printf("  LearningRate: %.e", flagLearningRate)
	log.Printf("  WeightDecay: %.e", flagWeightDecay)
	log.Printf("  EpsilonRMSNorm: %.e", flagEpsilonRMSNorm)
	log.Printf("  EpsilonAdamW: %.e", flagEpsilonAdamW)
	log.Printf("  GradientClipValue: %.2f", flagGradientClipValue)
	log.Printf("  HiddenSizes Slice: %v", hiddenSizes)
	log.Println("--------------------------------------------------")

	bpeDataPath := *bpeDataFlag
	modelDataPath := *modelDataFlag
	validationDataPath := *validationDataFlag // Get validation data path
	startEpoch := 0
	var err error
	needsModelTraining := *trainFlag

	log.Println("Status: Initializing...")

	bpe = NewBPE(BpeSpecialTokens)
	bpeIsReady := false

	// --- Loading or Initializing ---
	if *checkpointFlag != "" {
		var loadedSolver *SolverAdamW
		var loadedBPE *BPE
		startEpoch, model, loadedSolver, loadedBPE, err = loadCheckpoint(*checkpointFlag)
		if err != nil {
			log.Fatalf("FATAL: Failed to load checkpoint from %s: %v", *checkpointFlag, err)
		}
		solver = loadedSolver
		bpe = loadedBPE
		bpeIsReady = true

		log.Println("--- Effective Configuration (After Checkpoint Load) ---")
		log.Printf("  BPEVocabSize (Target): %d", flagBPEVocabSize) // Show original target flag value
		log.Printf("  BPEActualVocabSize: %d", bpeActualVocabSize) // Show actual loaded size
		log.Printf("  EmbeddingDimension: %d", embeddingDimension)
		log.Printf("  NumExperts: %d", numExperts)
		log.Printf("  TrainSeqLength: %d", seqLength)
		log.Printf("  BatchSize: %d", batchSize)
		log.Printf("  Epochs: %d", flagEpochs) // Shows total epochs from checkpoint config
		log.Printf("  MaxResponseLength: %d", flagMaxResponseLength)
		log.Printf("  LearningRate: %.e", solver.LR) // Show loaded LR from solver
		log.Printf("  WeightDecay: %.e", solver.WD)
		log.Printf("  EpsilonRMSNorm: %.e", flagEpsilonRMSNorm) // Updated from checkpoint config
		log.Printf("  EpsilonAdamW: %.e", solver.Eps)
		log.Printf("  GradientClipValue: %.2f", flagGradientClipValue) // Updated from checkpoint config
		log.Printf("  HiddenSizes Slice: %v", hiddenSizes)
		log.Println("--------------------------------------------------")

		if bpeDataPath != "" {
			log.Printf("Warning: -bpe-data ('%s') provided but ignored because a checkpoint was loaded.", bpeDataPath)
		}
		if needsModelTraining && modelDataPath == "" {
			log.Fatal("FATAL: Training requested (-train) but no model data file specified (-model-data).")
		}

	} else {
		// Initialize from scratch
		log.Println("No checkpoint specified. Initializing from scratch.")
		if bpeDataPath != "" {
			err = trainBPEFromFile(bpeDataPath)
			if err != nil { log.Fatalf("FATAL: Failed to train BPE tokenizer: %v", err) }
			bpeIsReady = true
		} else {
			log.Println("Warning: No -bpe-data provided and no checkpoint. BPE tokenizer will be empty.")
			// If training is requested, this will likely cause a fatal error later.
		}
		if needsModelTraining && modelDataPath == "" {
			log.Fatal("FATAL: Training requested (-train) but no model data file specified (-model-data).")
		}
	}

	// --- Prepare Validation Data (if path provided) ---
	// Do this *after* BPE is loaded/trained, but *before* model training starts
	if validationDataPath != "" {
		if !bpeIsReady {
			log.Fatal("FATAL: Cannot prepare validation data because BPE tokenizer was not loaded or trained.")
		}
		log.Println("Preparing validation data using the BPE tokenizer...")
		valDataReady, valDataErr := prepareValidationData(validationDataPath)
		if valDataErr != nil {
			log.Printf("Warning: Validation data preparation failed: %v. Proceeding without validation.", valDataErr)
			validationBatches = nil // Ensure it's nil if prep failed
		} else if !valDataReady {
			log.Println("Warning: Validation data preparation indicated no data ready. Proceeding without validation.")
			validationBatches = nil
		} else {
			log.Println("Validation data prepared successfully.")
		}
	} else {
		log.Println("Info: No -validation-data path provided. Skipping validation.")
	}


	// --- Prepare Model Data if needed for training ---
	if needsModelTraining {
		if !bpeIsReady {
			log.Fatal("FATAL: Cannot prepare model data because BPE tokenizer was not loaded or trained.")
		}
		if modelDataPath == "" {
			// This check is technically redundant due to earlier checks, but good for clarity
			log.Fatal("FATAL: Model training requested (-train) but -model-data path is missing.")
		}

		log.Println("Preparing model training data using the BPE tokenizer...")
		dataReady, dataErr := prepareModelData(modelDataPath)
		if dataErr != nil { log.Fatalf("FATAL: Model data preparation failed: %v", dataErr) }
		if !dataReady { log.Fatalf("FATAL: Model data preparation indicated failure (no batches created).") }
		log.Println("Model training data prepared successfully.")

		// Initialize model and optimizer only if NOT loading from checkpoint
		if *checkpointFlag == "" {
			if !bpeIsReady || bpeActualVocabSize == 0 {
				log.Fatal("FATAL: Cannot initialize model. BPE not ready or vocab size is zero.")
			}
			log.Println("Initializing new model and optimizer...")
			model = InitMoEGRU(bpeActualVocabSize, embeddingDimension, hiddenSizes, bpeActualVocabSize, numExperts)
			solver = NewSolverAdamW(flagLearningRate, 0.9, 0.999, flagEpsilonAdamW, flagWeightDecay)

			// Log parameter count for new model
			totalParams := 0
			if model != nil {
				keys := make([]string, 0, len(model))
				for k := range model { keys = append(keys, k) }
				sort.Strings(keys)
				for _, k := range keys { if m := model[k]; m != nil { totalParams += m.N * m.D } }
			}
			log.Printf("-------------------------------------")
			log.Printf("Total parameters for new model: %d", totalParams)
			log.Printf("-------------------------------------")
		}

		// --- Execute Training ---
		if startEpoch < flagEpochs {
			log.Printf("Proceeding with model training from epoch %d up to target epoch %d...", startEpoch, flagEpochs)
			err = trainGRUModel(startEpoch) // trainGRUModel now handles validation internally
			if err != nil { log.Fatalf("FATAL: Model training failed: %v", err) }
		} else {
			log.Printf("Loaded checkpoint is already at or beyond the target epoch (%d >= %d). No further training needed.", startEpoch, flagEpochs)
			trainingComplete = true
			// Run one final validation pass if validation data exists
			if len(validationBatches) > 0 {
				log.Println("Running final validation pass on loaded/trained model...")
				finalValLoss, valErr := calculateValidationLoss(model, validationBatches)
				if valErr != nil {
					log.Printf("Error during final validation pass: %v", valErr)
				} else {
					log.Printf("Final Validation Loss: %.4f", finalValLoss)
				}
			}
		}

	} else {
		// Not training the model
		if *checkpointFlag != "" {
			log.Println("Checkpoint loaded. Skipping model training as -train flag was not provided.")
			trainingComplete = true
			// Run validation if checkpoint loaded and validation data exists
			if len(validationBatches) > 0 {
				log.Println("Checkpoint loaded, running validation pass...")
				finalValLoss, valErr := calculateValidationLoss(model, validationBatches)
				if valErr != nil {
					log.Printf("Error during validation pass: %v", valErr)
				} else {
					log.Printf("Validation Loss (from loaded checkpoint): %.4f", finalValLoss)
				}
			}
		} else {
			log.Println("No checkpoint loaded and model training not requested (-train). Cannot proceed to chat.")
			trainingComplete = false // Explicitly set to false
		}
	}

	// --- Start Chat Interface ---
	if !trainingComplete {
		log.Fatal("FATAL: Model is not ready for chat. Ensure a checkpoint is loaded or training (-train with -bpe-data and -model-data) is completed successfully.")
	}
	if bpe == nil || model == nil {
		log.Fatal("FATAL: BPE or Model is nil before starting chat. This should not happen if trainingComplete is true.")
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

		botResponse, genErr := generateResponse(input, flagMaxResponseLength)
		if genErr != nil {
			log.Printf("Error during response generation: %v", genErr)
			fmt.Println("Bot: Sorry, an error occurred while generating the response.")
		} else {
			fmt.Printf("Bot: %s\n", botResponse)
		}
	}

	log.Println("\nGoodbye!")
}

// --- Helper Functions for Rand --- (Keep as is)
func randi(a, b int) int {
	if a >= b { return a }
	return rand.Intn(b-a) + a
}
func randf(a, b float64) float64 {
	if a >= b { return a }
	return rand.Float64()*(b-a) + a
}
func randn(mu, std float64) float64 {
	return rand.NormFloat64()*std + mu
}
func stringSliceToIntSlice(strs []string) ([]int, error) {
	ints := make([]int, len(strs))
	var err error
	for i, s := range strs {
		ints[i], err = strconv.Atoi(s)
		if err != nil {
			return nil, fmt.Errorf("error converting '%s' to int: %w", s, err)
		}
	}
	return ints, nil
}