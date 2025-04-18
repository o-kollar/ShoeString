package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io/ioutil"
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
	batches            [][]TrainingSample // Store data in batches
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
	Rank          int      `json:"rank"`
	MergedTokenID int      `json:"mergedTokenId"`
	Pair          []string `json:"-"` // derived, not saved directly by key
	Result        string   `json:"-"` // derived
	ID            int      `json:"-"` // derived
}

type BPESavedState struct {
	SpecialTokens []string             `json:"specialTokens"`
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
// ... (Mat struct, NewMat, NewRandMat, Zeros, Get, Set, GetCol, ZeroGrads, Clone) ...
// ... (Graph struct, NewGraph, Backward, addBackward) ...
// ... (Activation Functions: Tanh, Sigmoid, Relu, Gelu, applyActivation) ...
// ... (Matrix Operations: Add, Mul, Eltmul, AddBroadcastCol) ...
// ... (Other Graph Ops: Ones, OneMinus) ...
// ... (Lookup, CombineExperts, RMSNorm, Softmax, SoftmaxStandalone, StackCols) ...
// --- Matrix Definition ---
type Mat struct {
	N  int       // Number of rows (features/size)
	D  int       // Number of columns (batch size or 1)
	W  []float64 // Weights (row-major order: w[row*D + col])
	Dw []float64 // Gradients
}

// NewMat creates a new matrix initialized with zeros.
// N = rows (features), D = columns (batch size)
func NewMat(n, d int) *Mat {
	assert(n >= 0 && d >= 0, "Matrix dimensions must be non-negative")
	if n*d == 0 { // Handle case where either n or d is 0
		return &Mat{N: n, D: d, W: []float64{}, Dw: []float64{}}
	}
	w := make([]float64, n*d)
	dw := make([]float64, n*d)
	return &Mat{N: n, D: d, W: w, Dw: dw}
}

// NewRandMat creates a new matrix with random Gaussian values.
// D is typically 1 for parameter matrices, but can be > 1 for inputs/states
func NewRandMat(n, d int, mu, stddev float64) *Mat {
	m := NewMat(n, d)
	for i := range m.W {
		m.W[i] = rand.NormFloat64()*stddev + mu
	}
	return m
}

// Zeros creates a slice of zeros.
func Zeros(n int) []float64 {
	if n <= 0 {
		return []float64{}
	}
	return make([]float64, n)
}

// Get returns the value at (row, col).
func (m *Mat) Get(row, col int) float64 {
	assert(row >= 0 && row < m.N && col >= 0 && col < m.D, fmt.Sprintf("Mat.Get index (%d,%d) out of bounds for %dx%d matrix", row, col, m.N, m.D))
	ix := row*m.D + col
	return m.W[ix]
}

// Set sets the value at (row, col).
func (m *Mat) Set(row, col int, v float64) {
	assert(row >= 0 && row < m.N && col >= 0 && col < m.D, fmt.Sprintf("Mat.Set index (%d,%d) out of bounds for %dx%d matrix", row, col, m.N, m.D))
	ix := row*m.D + col
	m.W[ix] = v
}

// GetCol returns a specific column as a new [N x 1] matrix.
func (m *Mat) GetCol(col int) *Mat {
	assert(col >= 0 && col < m.D, fmt.Sprintf("Mat.GetCol index %d out of bounds for %dx%d matrix", col, m.N, m.D))
	colMat := NewMat(m.N, 1)
	for i := 0; i < m.N; i++ {
		colMat.W[i] = m.Get(i, col)
	}
	return colMat
}

// ZeroGrads sets all gradients to zero.
func (m *Mat) ZeroGrads() {
	for i := range m.Dw {
		m.Dw[i] = 0
	}
}

// Clone creates a deep copy of the matrix (including weights, not gradients).
func (m *Mat) Clone() *Mat {
	newM := NewMat(m.N, m.D)
	copy(newM.W, m.W)
	return newM
}

// --- Graph for Autograd (Mutex added previously) ---
type Graph struct {
	NeedsBackprop bool
	Backprop      []func() // Slice of functions to execute for backpropagation
	mu            sync.Mutex // Mutex to protect concurrent appends to Backprop
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

// Helper to safely append backward function
func (g *Graph) addBackward(f func()) {
	if g.NeedsBackprop {
		g.mu.Lock()
		g.Backprop = append(g.Backprop, f)
		g.mu.Unlock()
	}
}

// --- Activation Functions (Work element-wise, naturally handle batches) ---

// Constants for GELU
const (
	invSqrt2   = 0.7071067811865476 // 1.0 / math.Sqrt(2.0)
	invSqrt2pi = 0.3989422804014327 // 1.0 / math.Sqrt(2.0 * math.Pi)
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
		// derivative = 1 - tanh^2(x) = 1 - out^2
		return 1.0 - out_wi*out_wi
	})
}

func (g *Graph) Sigmoid(m *Mat) *Mat {
	sigmoid := func(x float64) float64 { return 1.0 / (1.0 + math.Exp(-x)) }
	derivative := func(m_wi, out_wi float64) float64 {
		// derivative = sigmoid(x) * (1 - sigmoid(x)) = out * (1 - out)
		return out_wi * (1.0 - out_wi)
	}
	return applyActivation(g, m, sigmoid, derivative)
}

func (g *Graph) Relu(m *Mat) *Mat {
	relu := func(x float64) float64 { return math.Max(0, x) }
	derivative := func(m_wi, out_wi float64) float64 {
		// derivative = 1 if x > 0, else 0
		if m_wi > 0 {
			return 1.0
		}
		return 0.0
	}
	return applyActivation(g, m, relu, derivative)
}

// Gelu implements the Gaussian Error Linear Unit activation.
func (g *Graph) Gelu(m *Mat) *Mat {
	// GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
	geluFunc := func(x float64) float64 {
		return 0.5 * x * (1.0 + math.Erf(x*invSqrt2))
	}

	// Derivative of GELU(x) = Φ(x) + x * φ(x)
	// where Φ(x) is the standard Gaussian CDF = 0.5 * (1 + erf(x / sqrt(2))) = GELU(x) / x (for x != 0)
	// and φ(x) is the standard Gaussian PDF = (1 / sqrt(2*pi)) * exp(-x^2 / 2)
	geluDerivative := func(x, gelu_x float64) float64 {
		phi_x := invSqrt2pi * math.Exp(-0.5*x*x)

		var phi_cap_x float64
		if math.Abs(x) < 1e-9 { // Avoid division by zero, Φ(0) = 0.5
			phi_cap_x = 0.5
		} else {
			phi_cap_x = gelu_x / x // Use precomputed GELU output (out_wi)
		}
		derivative := phi_cap_x + x*phi_x
		// Clamp derivative to avoid potential instability? Optional.
		// if math.IsNaN(derivative) || math.IsInf(derivative, 0) { return 0 }
		return derivative
	}

	return applyActivation(g, m, geluFunc, geluDerivative)
}

// --- Matrix Operations ---

// Add works element-wise, handles batches if dimensions match.
func (g *Graph) Add(m1, m2 *Mat) *Mat {
	assert(m1.N == m2.N && m1.D == m2.D,
		fmt.Sprintf("Add: Matrices must have the same size. m1: %dx%d, m2: %dx%d", m1.N, m1.D, m2.N, m2.D))
	out := NewMat(m1.N, m1.D)
	for i := range m1.W {
		out.W[i] = m1.W[i] + m2.W[i]
	}

	if g.NeedsBackprop {
		backward := func() {
			for i := range m1.W {
				m1.Dw[i] += out.Dw[i]
				m2.Dw[i] += out.Dw[i]
			}
		}
		g.addBackward(backward)
	}
	return out
}

// Multiply (Matrix Multiplication)
// Handles batches correctly: [N x K] * [K x BatchSize] -> [N x BatchSize]
func (g *Graph) Mul(m1, m2 *Mat) *Mat {
	assert(m1.D == m2.N,
		fmt.Sprintf("Mul: Matrix dimensions misaligned. m1: %dx%d, m2: %dx%d", m1.N, m1.D, m2.N, m2.D))

	n := m1.N
	k := m1.D
	batchSizeOut := m2.D
	out := NewMat(n, batchSizeOut)

	for i := 0; i < n; i++ {
		for j := 0; j < batchSizeOut; j++ {
			dot := 0.0
			for l := 0; l < k; l++ {
				dot += m1.W[i*k+l] * m2.W[l*batchSizeOut+j]
			}
			out.W[i*batchSizeOut+j] = dot
		}
	}

	if g.NeedsBackprop {
		backward := func() {
			for i := 0; i < n; i++ {
				for j := 0; j < batchSizeOut; j++ {
					gradOut := out.Dw[i*batchSizeOut+j]
					if gradOut == 0 {
						continue
					}
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

// Eltmul works element-wise, handles batches if dimensions match.
func (g *Graph) Eltmul(m1, m2 *Mat) *Mat {
	assert(m1.N == m2.N && m1.D == m2.D,
		fmt.Sprintf("Eltmul: Matrices must have the same size. m1: %dx%d, m2: %dx%d", m1.N, m1.D, m2.N, m2.D))
	out := NewMat(m1.N, m1.D)
	for i := range m1.W {
		out.W[i] = m1.W[i] * m2.W[i]
	}

	if g.NeedsBackprop {
		backward := func() {
			for i := range m1.W {
				m1.Dw[i] += m2.W[i] * out.Dw[i]
				m2.Dw[i] += m1.W[i] * out.Dw[i]
			}
		}
		g.addBackward(backward)
	}
	return out
}

// --- Add with Column Broadcasting ---

// AddBroadcastCol adds a column vector m2Col [N x 1] to each column of m1 [N x BatchSize].
func (g *Graph) AddBroadcastCol(m1 *Mat, m2Col *Mat) *Mat {
	assert(m1.N == m2Col.N, fmt.Sprintf("AddBroadcastCol: Row dimension mismatch. m1: %dx%d, m2Col: %dx%d", m1.N, m1.D, m2Col.N, m2Col.D))
	assert(m2Col.D == 1, fmt.Sprintf("AddBroadcastCol: m2Col must be a column vector (D=1), got %dx%d", m2Col.N, m2Col.D))

	n := m1.N
	batchSize := m1.D
	out := NewMat(n, batchSize)

	for j := 0; j < batchSize; j++ {
		for i := 0; i < n; i++ {
			out.W[i*batchSize+j] = m1.W[i*batchSize+j] + m2Col.W[i]
		}
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

// --- Other Graph Operations ---

// Ones creates a matrix filled with 1.0
func (g *Graph) Ones(n, d int) *Mat {
	m := NewMat(n, d)
	for i := range m.W {
		m.W[i] = 1.0
	}
	return m
}

// OneMinus computes 1 - m element-wise, handles batches.
func (g *Graph) OneMinus(m *Mat) *Mat {
	out := NewMat(m.N, m.D)
	for i := range m.W {
		out.W[i] = 1.0 - m.W[i]
	}

	if g.NeedsBackprop {
		backward := func() {
			for i := range m.W {
				m.Dw[i] += -1.0 * out.Dw[i]
			}
		}
		g.addBackward(backward)
	}
	return out
}

// --- Graph Operation: Lookup ---

// Lookup fetches embedding vectors for a batch of token IDs.
// Inputs:
//   - embeddingMatrix: The embedding table, *Mat of shape [VocabSize x EmbeddingDim].
//   - tokenIDs: A slice of integer token IDs for the batch. Length == BatchSize.
// Output:
//   - *Mat of shape [EmbeddingDim x BatchSize], where each column is the embedding for a token ID.
func (g *Graph) Lookup(embeddingMatrix *Mat, tokenIDs []int) *Mat {
	vocabSize := embeddingMatrix.N
	embeddingDim := embeddingMatrix.D
	batchSize := len(tokenIDs)

	assert(batchSize > 0, "Lookup: tokenIDs slice cannot be empty.")

	// --- Forward Pass ---
	out := NewMat(embeddingDim, batchSize)
	validIndices := make([]int, batchSize) // Store original indices of valid lookups for backprop

	for j, tokenID := range tokenIDs {
		validIndices[j] = tokenID
		if tokenID < 0 || tokenID >= vocabSize {
			// Handle invalid token ID - output zeros
			// log.Printf("Warning: Invalid token ID %d in Lookup batch item %d. Using zero vector.", tokenID, j) // Can be noisy
			validIndices[j] = -1 // Mark as invalid for backprop
			// No need to explicitly zero out.W, as NewMat initializes with zeros
			continue
		}

		// Copy the embedding vector (row `tokenID` from embeddingMatrix) to column `j` of `out`
		srcOffset := tokenID * embeddingDim
		destCol := j
		for i := 0; i < embeddingDim; i++ {
			out.W[i*batchSize+destCol] = embeddingMatrix.W[srcOffset+i]
		}
	}

	// --- Backward Pass Definition ---
	if g.NeedsBackprop {
		backward := func() {
			// Gradient dLoss/dOut is stored in out.Dw [EmbeddingDim x BatchSize]
			// Add these gradients back to the correct rows in embeddingMatrix.Dw

			for j := 0; j < batchSize; j++ {
				tokenID := validIndices[j]
				if tokenID == -1 {
					continue // Skip gradient accumulation for invalid tokens
				}

				targetRowOffset := tokenID * embeddingDim
				srcCol := j

				for i := 0; i < embeddingDim; i++ {
					grad := out.Dw[i*batchSize+srcCol]
					if !math.IsNaN(grad) && !math.IsInf(grad, 0) {
						embeddingMatrix.Dw[targetRowOffset+i] += grad
					}
				}
			}
		}
		g.addBackward(backward)
	}

	return out
}

// --- Graph Operation: CombineExperts ---

// CombineExperts calculates the weighted sum of expert outputs based on gating weights.
func (g *Graph) CombineExperts(expertOutputs []*Mat, gatingWeights *Mat) *Mat {
	if len(expertOutputs) == 0 {
		log.Panic("CombineExperts: expertOutputs slice cannot be empty.")
	}
	if gatingWeights == nil {
		log.Panic("CombineExperts: gatingWeights cannot be nil.")
	}

	numExperts := len(expertOutputs)
	hiddenSize := expertOutputs[0].N
	batchSize := expertOutputs[0].D

	// Validation
	assert(gatingWeights.N == numExperts, fmt.Sprintf("CombineExperts: gatingWeights rows (%d) must match numExperts (%d)", gatingWeights.N, numExperts))
	assert(gatingWeights.D == batchSize, fmt.Sprintf("CombineExperts: gatingWeights cols (%d) must match batch size (%d)", gatingWeights.D, batchSize))
	for e := 0; e < numExperts; e++ {
		assert(expertOutputs[e] != nil, fmt.Sprintf("CombineExperts: expertOutput %d is nil", e))
		assert(expertOutputs[e].N == hiddenSize, fmt.Sprintf("CombineExperts: expertOutput %d rows (%d) must match hiddenSize (%d)", e, expertOutputs[e].N, hiddenSize))
		assert(expertOutputs[e].D == batchSize, fmt.Sprintf("CombineExperts: expertOutput %d cols (%d) must match batch size (%d)", e, expertOutputs[e].D, batchSize))
	}

	// Forward Pass
	out := NewMat(hiddenSize, batchSize)
	for e := 0; e < numExperts; e++ {
		expertOut_e := expertOutputs[e]
		for j := 0; j < batchSize; j++ {
			gateWeight_ej := gatingWeights.Get(e, j)
			if gateWeight_ej == 0 { continue }
			outOffset := j
			expertOffset := j
			for i := 0; i < hiddenSize; i++ {
				out.W[i*batchSize+outOffset] += expertOut_e.W[i*batchSize+expertOffset] * gateWeight_ej
			}
		}
	}

	// Backward Pass Definition
	if g.NeedsBackprop {
		backward := func() {
			// Calculate dLoss/dGatingWeights
			for e := 0; e < numExperts; e++ {
				for j := 0; j < batchSize; j++ {
					gradAccumGating_ej := 0.0
					outOffset := j
					expertOffset := j
					for i := 0; i < hiddenSize; i++ {
						gradOut_ij := out.Dw[i*batchSize+outOffset]
						expertVal_eij := expertOutputs[e].W[i*batchSize+expertOffset]
						gradAccumGating_ej += gradOut_ij * expertVal_eij
					}
					gwOffset := e*batchSize + j
					if !math.IsNaN(gradAccumGating_ej) && !math.IsInf(gradAccumGating_ej, 0) {
						gatingWeights.Dw[gwOffset] += gradAccumGating_ej
					}
				}
			}

			// Calculate dLoss/dExpertOutput[e]
			for e := 0; e < numExperts; e++ {
				expertOutDw_e := expertOutputs[e].Dw
				for j := 0; j < batchSize; j++ {
					gateWeight_ej := gatingWeights.Get(e, j)
					if gateWeight_ej == 0 { continue }
					outOffset := j
					expertDwOffset := j
					for i := 0; i < hiddenSize; i++ {
						gradOut_ij := out.Dw[i*batchSize+outOffset]
						gradExpOut_eij := gradOut_ij * gateWeight_ej
						if !math.IsNaN(gradExpOut_eij) && !math.IsInf(gradExpOut_eij, 0) {
							expertOutDw_e[i*batchSize+expertDwOffset] += gradExpOut_eij
						}
					}
				}
			}
		}
		g.addBackward(backward)
	}
	return out
}

// --- Batch-Aware RMSNorm ---
// Uses flagEpsilonRMSNorm
func (g *Graph) RMSNorm(m, gain *Mat) *Mat {
	assert(gain.N == m.N, fmt.Sprintf("RMSNorm gain rows must match input rows. m: %dx%d, gain: %dx%d", m.N, m.D, gain.N, gain.D))
	assert(gain.D == 1, fmt.Sprintf("RMSNorm gain must be a column vector (D=1). Got %dx%d", gain.N, gain.D))

	n := m.N
	batchSize := m.D
	out := NewMat(n, batchSize)
	rmsPerCol := make([]float64, batchSize)
	invRMSPerCol := make([]float64, batchSize)
	mNorm := NewMat(n, batchSize)

	// Forward Pass
	for j := 0; j < batchSize; j++ {
		meanSq := 0.0
		for i := 0; i < n; i++ {
			val := m.Get(i, j)
			meanSq += val * val
		}
		meanSq /= float64(n)
		rmsPerCol[j] = math.Sqrt(meanSq + flagEpsilonRMSNorm) // Use flag value
		invRMSPerCol[j] = 1.0 / rmsPerCol[j]

		for i := 0; i < n; i++ {
			normVal := m.Get(i, j) * invRMSPerCol[j]
			mNorm.Set(i, j, normVal)
			out.Set(i, j, normVal*gain.W[i])
		}
	}

	// Backward Pass
	if g.NeedsBackprop {
		backward := func() {
			gainDwTemp := Zeros(n)
			for j := 0; j < batchSize; j++ {
				sumDNormMTimesNegNormM_j := 0.0
				dNormM_j := Zeros(n)

				for i := 0; i < n; i++ {
					dOut_ij := out.Dw[i*batchSize+j]
					mNorm_ij := mNorm.Get(i, j)
					gain_i := gain.W[i]

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
			for i := 0; i < n; i++ {
				gain.Dw[i] += gainDwTemp[i]
			}
		}
		g.addBackward(backward)
	}
	return out
}

// --- Batch-Aware Softmax ---
func (g *Graph) Softmax(m *Mat) *Mat {
	n := m.N
	batchSize := m.D
	out := NewMat(n, batchSize)

	// Forward Pass
	for j := 0; j < batchSize; j++ {
		maxVal := -math.MaxFloat64
		for i := 0; i < n; i++ {
			val := m.Get(i, j)
			if val > maxVal {
				maxVal = val
			}
		}

		sumExp := 0.0
		expValsCol := Zeros(n)
		for i := 0; i < n; i++ {
			expVal := math.Exp(m.Get(i, j) - maxVal)
			if math.IsNaN(expVal) || math.IsInf(expVal, 0) {
				expVal = 0
			}
			expValsCol[i] = expVal
			sumExp += expVal
		}

		invSumExp := 1.0 / (sumExp + 1e-9)
		if sumExp < 1e-9 {
			invSumExp = 1.0 / float64(n) // Uniform distribution if sum is too small
			for i := 0; i < n; i++ {
				out.Set(i, j, invSumExp)
			}
		} else {
			for i := 0; i < n; i++ {
				out.Set(i, j, expValsCol[i]*invSumExp)
			}
		}
	}

	// Backward Pass
	if g.NeedsBackprop {
		backward := func() {
			for j := 0; j < batchSize; j++ {
				dL_dOutput_j := Zeros(n)
				probs_j := Zeros(n)
				for i := 0; i < n; i++ {
					dL_dOutput_j[i] = out.Dw[i*batchSize+j]
					probs_j[i] = out.W[i*batchSize+j]
				}

				dotProd := 0.0
				for k := 0; k < n; k++ {
					// Check for NaNs/Infs before multiplying
					if !math.IsNaN(dL_dOutput_j[k]) && !math.IsInf(dL_dOutput_j[k], 0) && !math.IsNaN(probs_j[k]) && !math.IsInf(probs_j[k], 0) {
						dotProd += dL_dOutput_j[k] * probs_j[k]
					}
				}
				if math.IsNaN(dotProd) || math.IsInf(dotProd, 0) {
					dotProd = 0 // Sanitize dot product
				}

				for i := 0; i < n; i++ {
					prob_i := probs_j[i]
					dL_dOutput_i := dL_dOutput_j[i]

					// Check for NaNs/Infs in individual terms
					if math.IsNaN(prob_i) || math.IsInf(prob_i, 0) || math.IsNaN(dL_dOutput_i) || math.IsInf(dL_dOutput_i, 0) {
						continue // Skip gradient update for this element
					}

					// Jacobian of softmax: diag(p) - p * p^T
					// dL/dInput_i = sum_k (dL/dOutput_k * dOutput_k/dInput_i)
					//             = sum_k (dL/dOutput_k * (prob_i * (delta_ik - prob_k)))
					//             = prob_i * (dL/dOutput_i - sum_k(dL/dOutput_k * prob_k))
					//             = prob_i * (dL/dOutput_i - dotProd)
					gradInput_i := prob_i * (dL_dOutput_i - dotProd)

					// Final check for NaN/Inf in the calculated gradient
					if !math.IsNaN(gradInput_i) && !math.IsInf(gradInput_i, 0) {
						m.Dw[i*batchSize+j] += gradInput_i
					}
				}
			}
		}
		g.addBackward(backward)
	}
	return out
}

// SoftmaxStandalone - Batch Aware Version
func SoftmaxStandalone(m *Mat) *Mat {
	n := m.N
	batchSize := m.D
	out := NewMat(n, batchSize)

	for j := 0; j < batchSize; j++ {
		maxVal := -math.MaxFloat64
		for i := 0; i < n; i++ {
			val := m.Get(i, j)
			if val > maxVal {
				maxVal = val
			}
		}

		s := 0.0
		expValsCol := Zeros(n)
		for i := 0; i < n; i++ {
			expVal := math.Exp(m.Get(i, j) - maxVal)
			if math.IsNaN(expVal) || math.IsInf(expVal, 0) {
				expVal = 0
			}
			expValsCol[i] = expVal
			s += expVal
		}

		invS := 1.0 / (s + 1e-9)
		if s < 1e-9 {
			invS = 1.0 / float64(n) // Use uniform if sum is near zero
			for i := 0; i < n; i++ {
				out.Set(i, j, invS)
			}
		} else {
			for i := 0; i < n; i++ {
				out.Set(i, j, expValsCol[i]*invS)
			}
		}
	}
	return out
}

// StackCols - Kept for potential use, marked as possibly incorrect for batching
func StackCols(g *Graph, mats []*Mat) *Mat {
	if len(mats) == 0 {
		log.Panic("stackCols requires a non-empty array of matrices.")
	}
	n := mats[0].N
	numMats := len(mats)
	dOut := numMats

	for i := 0; i < numMats; i++ {
		assert(mats[i] != nil, fmt.Sprintf("stackCols: Matrix %d is nil.", i))
		assert(mats[i].N == n, fmt.Sprintf("stackCols: Matrix %d has height %d, expected %d.", i, mats[i].N, n))
		assert(mats[i].D == 1, fmt.Sprintf("stackCols: Matrix %d has width %d, expected 1.", i, mats[i].D))
	}

	out := NewMat(n, dOut)
	for j := 0; j < numMats; j++ { // Iterate through columns (matrices)
		for i := 0; i < n; i++ { // Iterate through rows
			out.W[i*dOut+j] = mats[j].W[i] // mats[j] is the j-th column vector
		}
	}

	if g.NeedsBackprop {
		backward := func() {
			for j := 0; j < numMats; j++ { // Iterate through original matrices (columns of output)
				for i := 0; i < n; i++ { // Iterate through rows
					// Gradient for W element i of matrix j comes from Dw element (i, j) of the output
					mats[j].Dw[i] += out.Dw[i*dOut+j]
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
// InitMoEGRU now uses global variables `embeddingDimension`, `hiddenSizes`, `numExperts` implicitly for logging and structure
// and `bpeActualVocabSize` for output layer size.
func InitMoEGRU(vocabSize int, embeddingDim int, hiddenSizes []int, outputSize int, numExperts int) map[string]*Mat {
	log.Printf("Initializing model parameters (Experts: %d)...", numExperts)
	model := make(map[string]*Mat)

	initStdDev := func(size int) float64 {
		if size > 0 {
			// Kaiming/He initialization factor for ReLU-like activations (GELU is similar)
			// return math.Sqrt(2.0 / float64(size))
			return 0.08 // Keep previous simpler init for now
		}
		return 0.08
	}

	// --- Embedding Layer ---
	log.Printf("Initializing Embedding Layer WE: %d x %d", vocabSize, embeddingDim)
	stdEmbed := 0.02 // Often smaller init for embeddings
	model["WE"] = NewRandMat(vocabSize, embeddingDim, 0, stdEmbed) // Shape [VocabSize x EmbeddingDim]

	// --- GRU Layers ---
	layerInputSize := embeddingDim // Input to first layer is now embedding dimension

	for d, hiddenSize := range hiddenSizes { // hiddenSizes slice is built from flags
		log.Printf("Layer %d: Input Size %d, Hidden Size %d", d, layerInputSize, hiddenSize)

		// Gating parameters
		stdGate := initStdDev(layerInputSize)
		model[fmt.Sprintf("Wg%d", d)] = NewRandMat(numExperts, layerInputSize, 0, stdGate)
		model[fmt.Sprintf("bg%d", d)] = NewMat(numExperts, 1) // Bias for gating logits

		// Expert parameters
		for e := 0; e < numExperts; e++ {
			stdX := initStdDev(layerInputSize) // Std dev based on input size
			stdH := initStdDev(hiddenSize)     // Std dev based on hidden size
			expertSuffix := fmt.Sprintf("_exp%d", e)

			// Update gate (z_t) parameters: sigmoid(Wzx * x + bz)
			// Wzx maps input [layerInputSize] -> hidden [hiddenSize]
			model[fmt.Sprintf("Wzx%d%s", d, expertSuffix)] = NewRandMat(hiddenSize, layerInputSize, 0, stdX)
			model[fmt.Sprintf("bz%d%s", d, expertSuffix)] = NewMat(hiddenSize, 1) // Bias for update gate

			// Candidate hidden state (h_cand) parameters: gelu(Whx * x + Whh * h_prev + bh)
			// Whx maps input [layerInputSize] -> hidden [hiddenSize]
			model[fmt.Sprintf("Whx%d%s", d, expertSuffix)] = NewRandMat(hiddenSize, layerInputSize, 0, stdX)
			// Whh maps hidden [hiddenSize] -> hidden [hiddenSize]
			model[fmt.Sprintf("Whh%d%s", d, expertSuffix)] = NewRandMat(hiddenSize, hiddenSize, 0, stdH)
			model[fmt.Sprintf("bh%d%s", d, expertSuffix)] = NewMat(hiddenSize, 1) // Bias for candidate state
		}

		// Residual Projection (if input and hidden sizes differ)
		if layerInputSize != hiddenSize {
			log.Printf("  Layer %d: Adding projection %dx%d for residual connection.", d, hiddenSize, layerInputSize)
			stdProj := initStdDev(layerInputSize)
			model[fmt.Sprintf("Wp%d", d)] = NewRandMat(hiddenSize, layerInputSize, 0, stdProj)
			model[fmt.Sprintf("bp%d", d)] = NewMat(hiddenSize, 1)
		} else {
			log.Printf("  Layer %d: Residual connection dimensions match (%dx%d), no projection needed.", d, hiddenSize, layerInputSize)
		}

		// RMSNorm Gain (applied after residual connection)
		log.Printf("  Layer %d: Adding RMSNorm gain parameter g_rms%d (%dx1).", d, d, hiddenSize)
		gRMSKey := fmt.Sprintf("g_rms%d", d)
		model[gRMSKey] = NewMat(hiddenSize, 1)
		for i := range model[gRMSKey].W {
			model[gRMSKey].W[i] = 1.0 // Initialize gain to 1
		}

		layerInputSize = hiddenSize // Output of this layer becomes input to the next
	}

	// --- Output Layer ---
	// The input to the output layer is the output of the last GRU layer (RMSNorm applied)
	finalHiddenSize := layerInputSize
	if len(hiddenSizes) > 0 {
		finalHiddenSize = hiddenSizes[len(hiddenSizes)-1]
	}
	// Use outputSize (== bpeActualVocabSize) passed to the function
	log.Printf("Initializing Output Layer Whd: %d x %d", outputSize, finalHiddenSize)
	stdDec := initStdDev(finalHiddenSize)
	model["Whd"] = NewRandMat(outputSize, finalHiddenSize, 0, stdDec) // Decode from final hidden state to vocab
	model["bd"] = NewMat(outputSize, 1)                                // Bias for output logits

	log.Println("Parameter Keys Initialized:", len(model))
	return model
}

// ForwardResult holds the outputs of the forward pass.
type ForwardResult struct {
	H [][]*Mat // Hidden states per layer, per expert [layer][expert][HiddenSize x BatchSize]
	O *Mat      // Output logits [VocabSize x BatchSize]
}

// ForwardMoEGRU remains the same logically, uses global bpeActualVocabSize for assertion
func ForwardMoEGRU(g *Graph, model map[string]*Mat, hiddenSizes []int, numExperts int, x *Mat, prevHiddenStates [][]*Mat) ForwardResult {

	currentBatchSize := x.D // Batch size from input embedding matrix

	// --- Initialize hidden states if necessary ---
	// Check if the structure is correct and dimensions match the current batch size
	needsInit := prevHiddenStates == nil || len(prevHiddenStates) != len(hiddenSizes)
	if !needsInit {
		for dChk := 0; dChk < len(hiddenSizes); dChk++ {
			if len(prevHiddenStates[dChk]) != numExperts {
				needsInit = true; break
			}
			if len(prevHiddenStates[dChk]) > 0 {
				// Check dimensions of the first expert's state in the layer
				if prevHiddenStates[dChk][0] == nil || prevHiddenStates[dChk][0].N != hiddenSizes[dChk] || prevHiddenStates[dChk][0].D != currentBatchSize {
					// log.Printf("Debug: Reinitializing hidden state L%d. Prev D=%d, Current D=%d", dChk, prevHiddenStates[dChk][0].D, currentBatchSize) // Debug log
					needsInit = true; break
				}
			} else { needsInit = true; break } // Layer exists but has no expert states
		}
	}

	if needsInit {
		// log.Printf("Debug: Initializing hidden states for batch size %d", currentBatchSize) // Debug log
		prevHiddenStates = make([][]*Mat, len(hiddenSizes))
		for dInit := 0; dInit < len(hiddenSizes); dInit++ {
			prevHiddenStates[dInit] = make([]*Mat, numExperts)
			for eInit := 0; eInit < numExperts; eInit++ {
				prevHiddenStates[dInit][eInit] = NewMat(hiddenSizes[dInit], currentBatchSize)
			}
		}
	}

	currentHiddenStatesLayers := make([][]*Mat, len(hiddenSizes))
	inputToLayer := x // Shape [EmbeddingDim x BatchSize]

	for d, hiddenSize := range hiddenSizes {
		layerInputSize := inputToLayer.N // Correctly gets EmbeddingDim for layer 0, HiddenSize for others
		expertOutputs := make([]*Mat, numExperts)
		currentLayerExpertStates := make([]*Mat, numExperts)
		residualSource := inputToLayer // Keep track of input for residual connection

		// --- Gating ---
		wgKey := fmt.Sprintf("Wg%d", d); bgKey := fmt.Sprintf("bg%d", d)
		Wg := model[wgKey]; bg := model[bgKey]
		assert(Wg != nil && bg != nil, fmt.Sprintf("Gating weights %s or %s not found", wgKey, bgKey))
		assert(Wg.D == layerInputSize, fmt.Sprintf("Wg dim mismatch layer %d. Wg.D=%d, layerInputSize=%d", d, Wg.D, layerInputSize))
		// Gating logits = Wg * input + bg (broadcast)
		gatingLogitsLinear := g.Mul(Wg, inputToLayer)             // [NumExperts x BatchSize]
		gatingLogits := g.AddBroadcastCol(gatingLogitsLinear, bg) // [NumExperts x BatchSize]
		// Gating weights = Softmax(gating logits)
		gatingWeights := g.Softmax(gatingLogits) // [NumExperts x BatchSize]
		assert(gatingWeights.N == numExperts && gatingWeights.D == currentBatchSize, fmt.Sprintf("Gating weights dim error layer %d", d))


		// --- Experts (Parallelized) ---
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

				// Update gate z_t = sigmoid(Wzx * x_t + bz)
				zLinear := g.Mul(Wzx_e, inputToLayer)             // [HiddenSize x BatchSize]
				z_t_e := g.Sigmoid(g.AddBroadcastCol(zLinear, bz_e)) // [HiddenSize x BatchSize]

				// Candidate hidden state h_cand = gelu(Whx * x_t + Whh * h_{t-1} + bh)
				termWhx := g.Mul(Whx_e, inputToLayer) // [HiddenSize x BatchSize]
				termWhh := g.Mul(Whh_e, hPrevExpert)  // [HiddenSize x BatchSize]
				hCandLinear := g.Add(termWhx, termWhh)
				hCandidate_e := g.Gelu(g.AddBroadcastCol(hCandLinear, bh_e)) // [HiddenSize x BatchSize] *** USE GELU HERE ***

				// New hidden state h_t = (1 - z_t) * h_{t-1} + z_t * h_cand
				oneMinusZ_e := g.OneMinus(z_t_e)                // [HiddenSize x BatchSize]
				term1_e := g.Eltmul(oneMinusZ_e, hPrevExpert) // [HiddenSize x BatchSize]
				term2_e := g.Eltmul(z_t_e, hCandidate_e)      // [HiddenSize x BatchSize]
				hNewExpert := g.Add(term1_e, term2_e)         // [HiddenSize x BatchSize]
				assert(hNewExpert.N == hiddenSize && hNewExpert.D == currentBatchSize, fmt.Sprintf("h_new_expert dim error L%d E%d", d, expertIdx))

				expertOutputs[expertIdx] = hNewExpert          // Store expert output for combining
				currentLayerExpertStates[expertIdx] = hNewExpert // Store for next timestep's prev state
			}(e)
		}
		wgExperts.Wait() // Wait for all experts to finish

		// --- Combine Experts ---
		// Weighted sum: sum(gating_weight_e * expert_output_e)
		hNewCombined := g.CombineExperts(expertOutputs, gatingWeights) // [HiddenSize x BatchSize]

		// --- Residual Connection ---
		var projectedResidual *Mat
		if layerInputSize == hiddenSize {
			// Dimensions match, use input directly
			projectedResidual = residualSource
		} else {
			// Dimensions differ, apply projection Wp * x + bp
			wpKey, bpKey := fmt.Sprintf("Wp%d", d), fmt.Sprintf("bp%d", d)
			Wp, bp := model[wpKey], model[bpKey]
			assert(Wp != nil && bp != nil, fmt.Sprintf("Projection Wp%d or bp%d not found.", d, d))
			projLinear := g.Mul(Wp, residualSource)                  // [HiddenSize x BatchSize]
			projectedResidual = g.AddBroadcastCol(projLinear, bp) // [HiddenSize x BatchSize]
		}
		assert(projectedResidual.N == hNewCombined.N && projectedResidual.D == hNewCombined.D, "Residual dim mismatch")
		outputWithResidual := g.Add(hNewCombined, projectedResidual) // [HiddenSize x BatchSize]

		// --- RMSNorm ---
		gRMSKey := fmt.Sprintf("g_rms%d", d)
		gRMS := model[gRMSKey]
		assert(gRMS != nil && gRMS.N == hiddenSize && gRMS.D == 1, fmt.Sprintf("RMSNorm gain g_rms%d error.", d))
		normalizedOutput := g.RMSNorm(outputWithResidual, gRMS) // [HiddenSize x BatchSize]
		assert(normalizedOutput.N == hiddenSize && normalizedOutput.D == currentBatchSize, fmt.Sprintf("RMSNorm output dim error L%d", d))

		// --- Prepare for next layer ---
		currentHiddenStatesLayers[d] = currentLayerExpertStates // Store the list of expert states for this layer
		inputToLayer = normalizedOutput                         // Output of this layer is input to the next
	} // End layer loop (d)

	// --- Output Layer ---
	lastLayerOutput := inputToLayer // Output of the final GRU layer (after RMSNorm)
	finalHiddenSize := lastLayerOutput.N
	Whd, bd := model["Whd"], model["bd"]
	assert(Whd != nil && bd != nil, "Output weights Whd or bd not found")
	assert(Whd.D == finalHiddenSize, fmt.Sprintf("Output Whd dim mismatch. Whd.D=%d, finalHiddenSize=%d", Whd.D, finalHiddenSize))
	// Output logits = Whd * last_layer_output + bd (broadcast)
	outputLogitsLinear := g.Mul(Whd, lastLayerOutput)         // [VocabSize x BatchSize]
	outputLogits := g.AddBroadcastCol(outputLogitsLinear, bd) // [VocabSize x BatchSize]
	// Use global bpeActualVocabSize for the assertion
	assert(outputLogits.N == bpeActualVocabSize && outputLogits.D == currentBatchSize, fmt.Sprintf("Output logits dim error. Got %dx%d, expected %dx%d", outputLogits.N, outputLogits.D, bpeActualVocabSize, currentBatchSize))

	return ForwardResult{H: currentHiddenStatesLayers, O: outputLogits}
}

//======================================================================
// --- Model Parameter Utilities --- (Keep as is)
//======================================================================
func GetModelParameters(model map[string]*Mat) []*Mat {
	params := make([]*Mat, 0, len(model))
	// Ensure consistent order for reproducibility if needed (maps iterate randomly)
	keys := make([]string, 0, len(model))
	for k := range model {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		params = append(params, model[k])
	}
	// Original simpler version (order not guaranteed):
	// for _, mat := range model {
	// 	params = append(params, mat)
	// }
	return params
}

func ZeroModelGrads(model map[string]*Mat) {
	for _, mat := range model {
		mat.ZeroGrads()
	}
}

//======================================================================
// --- AdamW Optimizer ---
//======================================================================
// Uses flagLearningRate, flagEpsilonAdamW, flagWeightDecay
type SolverAdamW struct {
	LR        float64            // Set from flagLearningRate on creation/load
	Beta1     float64            // Usually fixed (e.g., 0.9)
	Beta2     float64            // Usually fixed (e.g., 0.999)
	Eps       float64            // Set from flagEpsilonAdamW on creation/load
	WD        float64            // Set from flagWeightDecay on creation/load
	T         int
	M         map[string][]float64 // Momentum
	V         map[string][]float64 // Velocity
	paramKeys map[string]bool      // Track keys seen
}

// NewSolverAdamW now takes parameters (like LR, Eps, WD) which should come from flags
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

func (s *SolverAdamW) Step(model map[string]*Mat) {
	s.T++
	t := float64(s.T)
	// Bias correction terms
	beta1PowT := math.Pow(s.Beta1, t)
	beta2PowT := math.Pow(s.Beta2, t)
	lrT := s.LR * math.Sqrt(1.0-beta2PowT) / (1.0-beta1PowT) // Adjusted learning rate with bias correction

	// Use sorted keys for deterministic initialization and update order
	keys := make([]string, 0, len(model))
    for k := range model {
        keys = append(keys, k)
    }
    sort.Strings(keys)

	// Detect new parameters and initialize moments
	for _, k := range keys {
		p := model[k]
		if _, exists := s.paramKeys[k]; !exists {
			s.M[k] = Zeros(len(p.W))
			s.V[k] = Zeros(len(p.W))
			s.paramKeys[k] = true
			// log.Printf("Optimizer: Initializing moments for new key '%s'", k) // Optional: log new keys
		}
	}


	// Update parameters
	for _, k := range keys {
		p := model[k]
		mK, mExists := s.M[k]
		vK, vExists := s.V[k]

		// Double-check existence and size (should be guaranteed by init loop now, but good practice)
		if !mExists || !vExists || len(mK) != len(p.W) || len(vK) != len(p.W) {
			log.Printf("Error: Optimizer state mismatch for key %s. Reinitializing.", k)
			s.M[k] = Zeros(len(p.W))
			s.V[k] = Zeros(len(p.W))
			mK = s.M[k]
			vK = s.V[k]
			if !mExists || !vExists { s.paramKeys[k] = true } // Ensure tracked if re-initialized here
		}


		for i := range p.W {
			grad := p.Dw[i]
			if math.IsNaN(grad) || math.IsInf(grad, 0) {
				// log.Printf("Warning: NaN/Inf gradient detected for %s[%d]. Setting to 0.", k, i)
				grad = 0.0
				p.Dw[i] = 0.0 // Also zero the source gradient
			}

			// AdamW Update:
			// 1. Update biased first moment estimate
			mK[i] = s.Beta1*mK[i] + (1.0-s.Beta1)*grad
			// 2. Update biased second raw moment estimate
			vK[i] = s.Beta2*vK[i] + (1.0-s.Beta2)*(grad*grad)

			// Sanitize moments just in case (e.g., if grad^2 was huge but finite)
			if math.IsNaN(mK[i]) || math.IsInf(mK[i], 0) { mK[i] = 0 }
			if math.IsNaN(vK[i]) || math.IsInf(vK[i], 0) { vK[i] = 0 }

			// 3. Compute bias-corrected moment estimates (implicitly done via lrT)
			// mHat = mK[i] / (1.0 - beta1PowT)
			// vHat = vK[i] / (1.0 - beta2PowT)

			// 4. Compute update with Adam formula (using bias-corrected lrT)
			denom := math.Sqrt(vK[i]) + s.Eps // Use biased V here, bias correction is in lrT
			if denom == 0 {
				// log.Printf("Warning: Zero denominator in AdamW update for %s[%d]. Skipping update.", k, i)
				continue
			}
			update := lrT * mK[i] / denom

			if math.IsNaN(update) || math.IsInf(update, 0) {
				// log.Printf("Warning: NaN/Inf update calculated for %s[%d]. Skipping update.", k, i)
				continue
			}

			// 5. Apply Adam update
			p.W[i] -= update

			// 6. Apply decoupled weight decay directly to the weight
			//    (Note: The order vs Adam update matters slightly, common practice is decay first or separate)
			//    Let's apply it *after* the main Adam step for consistency with common libraries like PyTorch's AdamW.
			p.W[i] -= s.LR * s.WD * p.W[i] // Use s.LR (base LR) and s.WD here
		}
		// Reset gradient AFTER processing all weight updates for this parameter
		p.ZeroGrads()
		// No need to reassign slices s.M[k]=mK etc., as they are references
	}
}

// GetState extracts the serializable state of the optimizer.
func (s *SolverAdamW) GetState() SerializableSolverState {
	// Ensure all tracked keys have corresponding M and V entries before saving
	for key := range s.paramKeys {
		if _, exists := s.M[key]; !exists {
			log.Printf("Warning: Optimizer GetState detected missing M for key %s, initializing.", key)
			s.M[key] = Zeros(0) // Initialize empty, size will be checked on load
		}
		if _, exists := s.V[key]; !exists {
			log.Printf("Warning: Optimizer GetState detected missing V for key %s, initializing.", key)
			s.V[key] = Zeros(0) // Initialize empty
		}
	}
	return SerializableSolverState{
		LR:    s.LR,    // Save the current LR, Beta1, Beta2, Eps, WD
		Beta1: s.Beta1,
		Beta2: s.Beta2,
		Eps:   s.Eps,
		WD:    s.WD,
		T:     s.T,
		M:     s.M,
		V:     s.V,
	}
}

// LoadState configures the optimizer from a saved state.
func (s *SolverAdamW) LoadState(state SerializableSolverState) {
	s.LR = state.LR       // Load LR, Beta1, Beta2, Eps, WD from state
	s.Beta1 = state.Beta1
	s.Beta2 = state.Beta2
	s.Eps = state.Eps
	s.WD = state.WD
	s.T = state.T
	s.M = state.M
	s.V = state.V
	// Rebuild paramKeys from the loaded M map (assuming M and V have same keys)
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
		} else if tokenID != -1 { // -1 might be used for padding/masking
			log.Printf("Warning: Index %d out of bounds for one-hot vector size %d in batch item %d.", tokenID, vocabSize, j)
		}
	}
	return batchVec
}

//======================================================================
// --- BPE Training Function ---
//======================================================================
// trainBPEFromFile loads data from a path and trains the global bpe instance.
// Uses flagBPEVocabSize
func trainBPEFromFile(bpeDataPath string) error {
	if bpeDataPath == "" {
		return errors.New("BPE data path is empty")
	}
	if bpe == nil {
		return errors.New("global BPE instance is nil")
	}

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
	// Use the flagBPEVocabSize variable
	bpe.Train(bpeCorpus, flagBPEVocabSize, false, bpeLogWrapper)

	bpeActualVocabSize = len(bpe.vocabArray)
	if bpeActualVocabSize == 0 {
		return errors.New("BPE vocab size is zero after training")
	}
	log.Printf("BPE Actual Vocab Size after training: %d", bpeActualVocabSize)
	log.Println("Status: BPE training complete.")
	return nil
}

//======================================================================
// --- Model Data Preparation (Batching and Shuffling) ---
//======================================================================
// prepareModelData loads model training data, encodes it using the *existing* BPE,
// and creates batches. Uses global `seqLength` and `batchSize` (which are set from flags).
func prepareModelData(modelDataPath string) (bool, error) {
	log.Printf("Status: Preparing model training data from file '%s'...", modelDataPath)
	log.Println("\n--- Preparing Model Data ---")
	batches = [][]TrainingSample{} // Reset global batches

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

	// Encode the model text using the *already trained/loaded* BPE
	encodedTextIDs := bpe.Encode(modelText)
	log.Printf("Encoded model text -> %d tokens.", len(encodedTextIDs))

	// Check if text is long enough to create at least one sequence
	// Use global `seqLength` (set from flagTrainSeqLength)
	if len(encodedTextIDs) <= seqLength {
		err := fmt.Errorf("error: Encoded model text length (%d) is not greater than sequence length (%d). Cannot create training samples.", len(encodedTextIDs), seqLength)
		log.Println("Status: Error:", err)
		return false, err
	}

	// Create individual training samples (Input: t to t+seqLength-1, Target: t+1 to t+seqLength)
	allSamples := []TrainingSample{}
	for i := 0; i <= len(encodedTextIDs)-seqLength-1; i++ { // Adjusted loop bound using seqLength
		inputSeqIDs := encodedTextIDs[i : i+seqLength]
		targetSeqIDs := encodedTextIDs[i+1 : i+seqLength+1]
		allSamples = append(allSamples, TrainingSample{
			Input:  append([]int{}, inputSeqIDs...),  // Create copies
			Target: append([]int{}, targetSeqIDs...), // Create copies
		})
	}

	log.Println("Total individual sequences generated:", len(allSamples))
	if len(allSamples) == 0 {
		// This should theoretically not happen if len(encodedTextIDs) > seqLength
		return false, errors.New("no training sequences generated despite sufficient text length")
	}

	// Adjust batch size if we have fewer samples than the configured batch size
	// Use global `batchSize` (set from flagBatchSize)
	currentBatchSize := batchSize
	if len(allSamples) < currentBatchSize {
		log.Printf("Warning: Number of samples (%d) is less than configured batch size (%d). Adjusting batch size for this run.", len(allSamples), currentBatchSize)
		currentBatchSize = len(allSamples) // Adjust for this run
		if currentBatchSize == 0 {
			return false, errors.New("adjusted batch size became zero (no samples)")
		}
	}

	// Shuffle the samples
	rand.Shuffle(len(allSamples), func(i, j int) {
		allSamples[i], allSamples[j] = allSamples[j], allSamples[i]
	})
	log.Println("Shuffled training samples.")

	// Create batches
	numBatches := len(allSamples) / currentBatchSize
	batches = make([][]TrainingSample, 0, numBatches) // Initialize global batches slice
	for i := 0; i < numBatches; i++ {
		start := i * currentBatchSize
		end := start + currentBatchSize
		batch := allSamples[start:end]
		batches = append(batches, append([]TrainingSample{}, batch...)) // Append a copy of the batch slice
	}

	leftoverCount := len(allSamples) % currentBatchSize
	if leftoverCount > 0 {
		log.Printf("Warning: Discarding %d leftover samples that don't form a full batch.", leftoverCount)
	}

	log.Printf("Created %d batches of size %d.", len(batches), currentBatchSize)
	if len(batches) == 0 {
		// This could happen if e.g., samples < original batchSize, and adjusted batchSize became 0? Should be caught earlier.
		// Or if numBatches calculation resulted in 0.
		return false, errors.New("no batches created")
	}

	log.Println("Status: Model data preparation complete.")
	return true, nil
}


//======================================================================
// --- Checkpointing Structures and Functions ---
//======================================================================
// SerializableMat remains the same
type SerializableMat struct {
	N int       `json:"n"`
	D int       `json:"d"`
	W []float64 `json:"w"`
	// Dw is usually not saved for inference, but needed for resuming training.
	Dw []float64 `json:"dw,omitempty"` // Use omitempty if we decide not to save Dw always
}

// SerializableSolverState remains the same
type SerializableSolverState struct {
	LR    float64              `json:"lr"`
	Beta1 float64              `json:"beta1"`
	Beta2 float64              `json:"beta2"`
	Eps   float64              `json:"eps"`
	WD    float64              `json:"wd"`
	T     int                  `json:"t"`
	M     map[string][]float64 `json:"m"`
	V     map[string][]float64 `json:"v"`
}

// Checkpoint struct now includes the flag values used during saving
type Checkpoint struct {
	Epoch          int                       `json:"epoch"`
	ModelParams    map[string]SerializableMat `json:"model_params"`
	OptimizerState SerializableSolverState   `json:"optimizer_state"`
	BPEState       BPESavedState             `json:"bpe_state"`
	// Store config for validation on load - use the flag variable names
	Config struct {
		BPEVocabSize       int     `json:"bpe_vocab_size"` // Target size from flag
		EmbeddingDimension int     `json:"embedding_dimension"`
		GRUHiddenSize      int     `json:"gru_hidden_size"`
		GRULayers          int     `json:"gru_layers"`
		NumExperts         int     `json:"num_experts"`
		TrainSeqLength     int     `json:"train_seq_length"`
		BatchSize          int     `json:"batch_size"`
		Epochs             int     `json:"epochs"` // Total epochs config from flag
		MaxResponseLength  int     `json:"max_response_length"`
		LearningRate       float64 `json:"learning_rate"`
		WeightDecay        float64 `json:"weight_decay"`
		EpsilonRMSNorm     float64 `json:"epsilon_rmsnorm"`
		EpsilonAdamW       float64 `json:"epsilon_adamw"`
		GradientClipValue  float64 `json:"gradient_clip_value"`
		// Store derived/runtime values as well
		BPEActualVocabSize int   `json:"bpe_actual_vocab_size"` // Actual used size
		HiddenSizes        []int `json:"hidden_sizes"` // Specific hidden sizes per layer (derived)
	} `json:"config"`
}

// matToSerializable remains the same
func matToSerializable(m *Mat) SerializableMat {
	// Create copies of slices to avoid modification issues if original Mat is changed later
	wCopy := make([]float64, len(m.W))
	dwCopy := make([]float64, len(m.Dw))
	copy(wCopy, m.W)
	copy(dwCopy, m.Dw)
	return SerializableMat{
		N:  m.N,
		D:  m.D,
		W:  wCopy,
		Dw: dwCopy, // Include gradients for resuming training
	}
}

// serializableToMat remains the same
func serializableToMat(sm SerializableMat) *Mat {
	m := NewMat(sm.N, sm.D)
	// Important: Copy slices to avoid aliasing issues if sm is reused
	copy(m.W, sm.W)
	if len(sm.Dw) == len(m.Dw) { // Only copy Dw if present and correct size
		copy(m.Dw, sm.Dw)
	} else if len(sm.Dw) != 0 {
		log.Printf("Warning: Checkpoint Dw size (%d) mismatch for matrix %dx%d (expected %d), gradients not loaded.", len(sm.Dw), sm.N, sm.D, len(m.Dw))
	}
	return m
}

// saveCheckpoint saves the current training state, including the flag values.
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
		Epoch:          epoch, // Save the epoch that just finished
		ModelParams:    serializableModel,
		OptimizerState: optimizerState,
		BPEState:       bpeState,
	}
	// Add config values *from the flag variables* at the time of saving
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

	// Add derived/runtime values
	checkpoint.Config.BPEActualVocabSize = bpeActualVocabSize // Use the actual vocab size
	checkpoint.Config.HiddenSizes = append([]int{}, hiddenSizes...) // Copy slice

	// 5. Marshal to JSON
	data, err := json.MarshalIndent(checkpoint, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal checkpoint data: %w", err)
	}

	// 6. Write to file atomically (write to temp, then rename)
	tempPath := path + ".tmp"
	err = ioutil.WriteFile(tempPath, data, 0644)
	if err != nil {
		return fmt.Errorf("failed to write temporary checkpoint file %s: %w", tempPath, err)
	}
	err = os.Rename(tempPath, path)
	if err != nil {
		_ = os.Remove(tempPath)
		return fmt.Errorf("failed to rename temporary checkpoint file to %s: %w", path, err)
	}

	log.Printf("Checkpoint saved successfully to %s", path)
	return nil
}

// loadCheckpoint loads training state and updates global config vars if loading only checkpoint.
func loadCheckpoint(path string) (startEpoch int, loadedModel map[string]*Mat, loadedSolver *SolverAdamW, loadedBPE *BPE, err error) {
	log.Printf("Loading checkpoint from %s...", path)

	// 1. Read file
	data, err := ioutil.ReadFile(path)
	if err != nil {
		err = fmt.Errorf("failed to read checkpoint file %s: %w", path, err)
		return
	}

	// 2. Unmarshal JSON
	var checkpoint Checkpoint
	err = json.Unmarshal(data, &checkpoint)
	if err != nil {
		err = fmt.Errorf("failed to unmarshal checkpoint data from %s: %w", path, err)
		return
	}

	// 3. Validate Config against current flag values (informational)
	log.Println("Validating checkpoint configuration against current flag settings...")
	configMismatch := false
	// Compare only key architectural/training parameters where mismatch is critical or needs warning
	if checkpoint.Config.EmbeddingDimension != flagEmbeddingDimension {
		log.Printf("Warning: Checkpoint EmbeddingDimension (%d) differs from current flag (%d)", checkpoint.Config.EmbeddingDimension, flagEmbeddingDimension); configMismatch=true
	}
	if len(checkpoint.Config.HiddenSizes) != flagGRULayers {
		log.Printf("Warning: Checkpoint GRULayers based on HiddenSizes length (%d) differs from current flag (%d)", len(checkpoint.Config.HiddenSizes), flagGRULayers); configMismatch=true
	} else {
		// Check individual hidden sizes only if layer count matches
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
	// BPE Actual size is important, but no flag to compare against directly. Just log it.
	log.Printf("Info: Checkpoint BPE Actual Vocab Size: %d", checkpoint.Config.BPEActualVocabSize)
	// Optimizer params are loaded separately, but warn about LR/WD mismatch
	if math.Abs(checkpoint.Config.LearningRate-flagLearningRate) > 1e-9 { // Compare floats carefully
		log.Printf("Warning: Checkpoint LearningRate (%.e) differs from current flag (%.e)", checkpoint.Config.LearningRate, flagLearningRate); configMismatch=true
	}
	if math.Abs(checkpoint.Config.WeightDecay-flagWeightDecay) > 1e-9 {
		log.Printf("Warning: Checkpoint WeightDecay (%.e) differs from current flag (%.e)", checkpoint.Config.WeightDecay, flagWeightDecay); configMismatch=true
	}
	// Others like batch size, epochs, epsilons, clip value are less critical to warn about if they differ,
	// as the loaded optimizer state or current flags will take precedence depending on usage context.

	if configMismatch {
		log.Println("Configuration mismatch detected. Checkpoint values will be used for model structure and optimizer state. Current flags may affect subsequent training or behavior if applicable.")
	} else {
		log.Println("Checkpoint configuration broadly matches current flag settings.")
	}

	// 4. Reconstruct Model
	loadedModel = make(map[string]*Mat)
	for k, sm := range checkpoint.ModelParams {
		loadedModel[k] = serializableToMat(sm)
		// TODO: Add validation of matrix sizes against expected sizes based on *loaded* config?
	}
	log.Printf("Loaded %d model parameters.", len(loadedModel))

	// 5. Reconstruct Optimizer (using state saved in checkpoint)
	loadedSolver = NewSolverAdamW(
		checkpoint.OptimizerState.LR, // Use LR/WD etc. *from the checkpoint state*
		checkpoint.OptimizerState.Beta1,
		checkpoint.OptimizerState.Beta2,
		checkpoint.OptimizerState.Eps,
		checkpoint.OptimizerState.WD,
	)
	loadedSolver.LoadState(checkpoint.OptimizerState) // Load M, V, T

	// 6. Reconstruct BPE Tokenizer
	loadedBPE = NewBPE(checkpoint.BPEState.SpecialTokens)
	err = loadedBPE.LoadState(checkpoint.BPEState)
	if err != nil {
		err = fmt.Errorf("failed to load BPE state from checkpoint: %w", err)
		return // Return error immediately if BPE fails
	}
	// *** Crucially, update the global `bpeActualVocabSize` based on the loaded BPE ***
	bpeActualVocabSize = len(loadedBPE.vocabArray)
	log.Printf("Loaded BPE tokenizer with %d vocab size.", bpeActualVocabSize)

	// 7. *** Update Global Configuration Variables from Checkpoint Config ***
	// This ensures that subsequent operations (like model init check, training loop setup, generation)
	// use the configuration the model was actually saved with, overriding command-line defaults
	// if only -checkpoint was provided.
	log.Println("Applying checkpoint configuration to runtime variables...")
	flagBPEVocabSize = checkpoint.Config.BPEVocabSize // Update flag vars themselves for consistency? Or just the derived globals? Let's update derived globals.
	embeddingDimension = checkpoint.Config.EmbeddingDimension
	hiddenSizes = append([]int{}, checkpoint.Config.HiddenSizes...) // Use the saved slice directly
	numExperts = checkpoint.Config.NumExperts
	seqLength = checkpoint.Config.TrainSeqLength
	batchSize = checkpoint.Config.BatchSize // Update batchSize from checkpoint
	flagMaxResponseLength = checkpoint.Config.MaxResponseLength // Update max response length
	// Update derived layer/hidden size flags if needed for consistency elsewhere, though hiddenSizes slice is primary now
	flagGRULayers = len(hiddenSizes)
	if len(hiddenSizes) > 0 {
		flagGRUHiddenSize = hiddenSizes[0] // Assume uniform hidden size for simplicity if needed
	}
	// Also update optimizer related flag variables to match loaded state for reference? Less critical as solver holds the state.
	flagLearningRate = loadedSolver.LR
	flagWeightDecay = loadedSolver.WD
	flagEpsilonAdamW = loadedSolver.Eps
	flagEpsilonRMSNorm = checkpoint.Config.EpsilonRMSNorm // Load this from checkpoint config
	flagGradientClipValue = checkpoint.Config.GradientClipValue
	flagEpochs = checkpoint.Config.Epochs // Update total epochs from checkpoint

	// 8. Return loaded state
	startEpoch = checkpoint.Epoch + 1 // Start training from the *next* epoch
	log.Printf("Checkpoint loaded successfully. Configuration updated. Resuming from epoch %d.", startEpoch)
	return // Returns named return values (startEpoch, loadedModel, loadedSolver, loadedBPE, err=nil)
}


//======================================================================
// --- Training Loop ---
//======================================================================
// Uses flagEpochs, seqLength, batchSize, embeddingDimension, flagGradientClipValue
func trainGRUModel(startEpoch int) error {
	// Ensure model and solver are initialized before calling this function
	if model == nil || solver == nil || bpe == nil {
		return errors.New("training called but model, solver, or BPE is not initialized")
	}
	if bpeActualVocabSize <= 0 {
	    return errors.New("training called but BPE vocab size is zero")
	}


	log.Printf("Status: starting from epoch %d...", startEpoch)
	log.Println("\n--- Training model ---")

	totalBatches := len(batches)
	if totalBatches == 0 {
		return errors.New("no batches found for training")
	}
	// Use global batchSize and embeddingDimension (set from flags/checkpoint)
	log.Printf("Starting training: %d total epochs configured, %d batches/epoch, Batch Size: %d, Embedding Dim: %d...", flagEpochs, totalBatches, batchSize, embeddingDimension)

	// Use flagEpochs for the loop limit
	for epoch := startEpoch; epoch < flagEpochs; epoch++ {
		log.Printf("Status: Starting Epoch %d/%d", epoch+1, flagEpochs)
		epochStartTime := time.Now()
		cumulativeEpochLoss := 0.0
		totalValidStepsInEpoch := 0 // Count valid (non-masked) loss calculations

		// Shuffle batches each epoch
		rand.Shuffle(len(batches), func(i, j int) {
			batches[i], batches[j] = batches[j], batches[i]
		})

		// Calculate progress interval based on total batches, ensuring it's at least 1
		progressInterval := totalBatches / 20
		if progressInterval == 0 { progressInterval = 1 }

		for batchIndex, batch := range batches {
			currentBatchSize := len(batch)
			if currentBatchSize == 0 {
				log.Printf("Warning: Skipping empty batch at index %d", batchIndex)
				continue
			}

			g := NewGraph(true)        // Create a new graph for each batch
			var hiddenStates [][]*Mat // Reset hidden state for each batch sequence
			batchLoss := 0.0
			validStepsInBatch := 0 // Count valid loss calculations within this batch

			// --- Process sequence within the batch ---
			// Use global seqLength (set from flag/checkpoint)
			for t := 0; t < seqLength; t++ {
				// 1. Prepare input batch token IDs and target IDs for this timestep
				inputTokenIDs := make([]int, currentBatchSize)
				targetTokenIDs := make([]int, currentBatchSize)
				hasValidTargetInStep := false // Track if any item in this step has a valid target

				for i := 0; i < currentBatchSize; i++ {
					// Ensure we don't go out of bounds for the sample's sequences
					if t < len(batch[i].Input) && t < len(batch[i].Target) {
						// Input ID: Use -1 for invalid/out-of-vocab, Lookup handles it
						if batch[i].Input[t] >= 0 && batch[i].Input[t] < bpeActualVocabSize {
							inputTokenIDs[i] = batch[i].Input[t]
						} else {
							// log.Printf("Debug: Invalid input token %d at batch %d item %d step %d", batch[i].Input[t], batchIndex, i, t)
							inputTokenIDs[i] = -1 // Map out-of-bounds to -1 (Lookup treats as zero vector)
						}
						// Target ID: Use -1 for invalid/out-of-vocab, needed for loss masking
						if batch[i].Target[t] >= 0 && batch[i].Target[t] < bpeActualVocabSize {
							targetTokenIDs[i] = batch[i].Target[t]
							hasValidTargetInStep = true // Mark step as having at least one valid target
						} else {
							// log.Printf("Debug: Invalid target token %d at batch %d item %d step %d", batch[i].Target[t], batchIndex, i, t)
							targetTokenIDs[i] = -1 // Mark for masking
						}
					} else {
						// This case should ideally not happen if data prep ensures seqLength consistency
						inputTokenIDs[i] = -1
						targetTokenIDs[i] = -1
						// log.Printf("Warning: Accessing beyond sequence length at batch %d item %d step %d", batchIndex, i, t)
					}
				}

				// If no item in the batch has a valid target at this timestep, skip computation
				if !hasValidTargetInStep {
					continue // Skip forward/backward pass for this step
				}

				// 2. Get embeddings for the input tokens
				xBatch := g.Lookup(model["WE"], inputTokenIDs) // Shape [EmbeddingDim x BatchSize]

				// 3. Perform forward pass (uses global hiddenSizes, numExperts)
				forwardResult := ForwardMoEGRU(g, model, hiddenSizes, numExperts, xBatch, hiddenStates)
				hiddenStates = forwardResult.H // Update hidden states for the next step
				outputLogits := forwardResult.O // Shape [VocabSize x BatchSize]

				// 4. Calculate probabilities (standalone, no graph needed here)
				probs := SoftmaxStandalone(outputLogits) // Shape [VocabSize x BatchSize]

				// 5. Loss Calculation & Gradient Preparation (Masked by valid targets)
				stepLoss := 0.0
				// Initialize gradient matrix for this step's contribution to dLoss/dLogits
				// Gradients only accumulate for items with valid targets.
				dLdLogits := NewMat(bpeActualVocabSize, currentBatchSize) // Init grads to zero
				numValidInStep := 0

				for j := 0; j < currentBatchSize; j++ { // Iterate over items in the batch
					targetTokenID := targetTokenIDs[j]
					if targetTokenID == -1 { // Skip loss/gradient for items with invalid targets
						continue
					}

					// Get the probability of the target token for this item
					targetProb := probs.Get(targetTokenID, j)
					// Clamp probability to avoid log(0) -> NaN/Inf
					loss := -math.Log(math.Max(targetProb, 1e-9))

					if math.IsNaN(loss) || math.IsInf(loss, 0) {
						log.Printf("Warn: NaN/Inf loss Ep %d, Batch %d, Item %d, Step %d. TargetID: %d, Prob: %.4e. Skipping item.", epoch+1, batchIndex, j, t, targetTokenID, targetProb)
						continue // Skip this item's contribution to loss and gradient
					}

					numValidInStep++
					stepLoss += loss

					// Calculate gradient dL/dLogits = P - Y (where Y is one-hot for the target)
					// This gradient is only calculated *for this specific item j*.
					for i := 0; i < bpeActualVocabSize; i++ { // Iterate over vocab dimension
						delta := probs.Get(i, j)
						if i == targetTokenID {
							delta -= 1.0 // Subtract 1 for the target class
						}
						// Check for NaN/Inf in delta before setting gradient
						if !math.IsNaN(delta) && !math.IsInf(delta, 0) {
							dLdLogits.Set(i, j, delta) // Set gradient dL/dlogit_i for item j
						} else {
							dLdLogits.Set(i, j, 0.0) // Zero out if calculation resulted in NaN/Inf
						}
					}
				} // End loop over batch items (j)

				// Accumulate step loss and gradients *if* there were valid items
				if numValidInStep > 0 {
					batchLoss += stepLoss
					validStepsInBatch += numValidInStep // Accumulate count of valid steps in batch

					// Apply the calculated gradients (dLdLogits) to the outputLogits.Dw
					// Scale the gradient by the number of valid items in this step to get the average gradient
					scaleFactor := 1.0 / float64(numValidInStep)
					for j := 0; j < currentBatchSize; j++ {
						if targetTokenIDs[j] != -1 { // Check again if target was valid for this item
							for i := 0; i < bpeActualVocabSize; i++ {
								grad_ij := dLdLogits.Get(i, j) // This is 0 if target was invalid
								// Add the scaled gradient to the graph node's gradient accumulator
								outputLogits.Dw[i*currentBatchSize+j] += grad_ij * scaleFactor
							}
						}
					}
				}
			} // End sequence loop (t)

			// --- After processing all timesteps for the batch ---
			// Only perform backprop and optimizer step if there was valid loss contribution
			if validStepsInBatch > 0 && !math.IsNaN(batchLoss) && !math.IsInf(batchLoss, 0) {
				g.Backward() // Backpropagate accumulated gradients through the graph

				// Gradient Clipping (applied to all parameters in the model)
				// Use flagGradientClipValue
				params := GetModelParameters(model)
				var gradNormSq float64 = 0
				for _, p := range params {
					for _, dwVal := range p.Dw {
						if !math.IsNaN(dwVal) && !math.IsInf(dwVal, 0) {
							gradNormSq += dwVal * dwVal
						}
					}
				}

				// Check if gradient norm is valid before clipping/stepping
				if !math.IsNaN(gradNormSq) && !math.IsInf(gradNormSq, 0) && gradNormSq > 0 {
					gradNorm := math.Sqrt(gradNormSq)
					if gradNorm > flagGradientClipValue { // Use flag value
						scale := flagGradientClipValue / (gradNorm + 1e-7) // Add epsilon for stability
						for _, p := range params {
							for i := range p.Dw {
								// Only scale valid gradients
								if !math.IsNaN(p.Dw[i]) && !math.IsInf(p.Dw[i], 0) {
									p.Dw[i] *= scale
								} else {
									p.Dw[i] = 0 // Zero out invalid gradients discovered during norm calculation
								}
							}
						}
					}
					// Apply AdamW optimizer step (includes zeroing gradients internally now)
					solver.Step(model)
				} else {
					log.Printf("Warn: Grad norm invalid (%.4f) or zero Ep %d Batch %d. Zeroing grads and skipping step.", gradNormSq, epoch+1, batchIndex)
					ZeroModelGrads(model) // Zero grads, but don't step optimizer
				}

				// Accumulate total loss and valid steps for epoch average calculation
				cumulativeEpochLoss += batchLoss
				totalValidStepsInEpoch += validStepsInBatch

			} else if validStepsInBatch > 0 {
				log.Printf("Warn: Invalid batch loss (%.4f) despite %d valid steps Ep %d Batch %d. Zeroing grads.", batchLoss, validStepsInBatch, epoch+1, batchIndex)
				ZeroModelGrads(model) // Zero grads if batch loss calculation failed
			} // If validStepsInBatch == 0, no loss was calculated, so no backprop/step needed.

			// --- Progress Indicator ---
			if (batchIndex+1)%progressInterval == 0 || batchIndex == totalBatches-1 {
				// Update progress bar more frequently or at the end
				doneCount := batchIndex + 1
				percentage := float64(doneCount) / float64(totalBatches) * 100
				barLength := 20
				filledLength := int(percentage / 100 * float64(barLength))
				if filledLength > barLength { filledLength = barLength }
				if filledLength < 0 { filledLength = 0}
				bar := strings.Repeat("=", filledLength) + strings.Repeat("-", barLength-filledLength)
				// Use flagEpochs in display
				fmt.Printf("\rEpoch %d/%d [%s] %d/%d (%.1f%%)", epoch+1, flagEpochs, bar, doneCount, totalBatches, percentage)
			}

		} // End batch loop (batchIndex)
		fmt.Println() // Newline after progress bar completes

		// --- Epoch Summary ---
		avgEpochLoss := 0.0
		if totalValidStepsInEpoch > 0 {
			// Average loss per valid step across the epoch
			avgEpochLoss = cumulativeEpochLoss / float64(totalValidStepsInEpoch)
		} else {
			log.Printf("Warning: Epoch %d completed with zero valid steps.", epoch+1)
		}
		epochDuration := time.Since(epochStartTime)
		// Use flagEpochs in display
		log.Printf("Epoch: %d/%d, Average Step Loss: %.4f, Duration: %s", epoch+1, flagEpochs, avgEpochLoss, epochDuration)

		// --- Save Checkpoint ---
		checkpointFilename := fmt.Sprintf("checkpoint_epoch_%d.json", epoch)
		checkpointFilepath := filepath.Join(CheckpointDir, checkpointFilename)
		err := saveCheckpoint(epoch, model, solver, bpe, checkpointFilepath)
		if err != nil {
			log.Printf("Error saving checkpoint for epoch %d: %v", epoch, err)
			// Decide whether to continue training or stop if checkpoint fails.
			// For now, log the error and continue.
		}

	} // End Epoch Loop

	log.Println("--- Training Complete ---")
	log.Println("Status: Training finished. Ready for chat.")
	trainingComplete = true // Mark training as complete
	return nil
}


//======================================================================
// --- Conversational Response Generation ---
//======================================================================
// Uses flagMaxResponseLength
func generateResponse(inputText string, maxLength int) (string, error) {
	if !trainingComplete || bpe == nil || model == nil {
		return "Sorry, the model hasn't been trained or loaded yet.", nil
	}
	// Use global numExperts (set from flag/checkpoint)
	if numExperts <= 0 {
		return "Error: Model configuration issue (numExperts invalid).", errors.New("numExperts invalid")
	}
	if _, ok := model["WE"]; !ok {
		return "Error: Model configuration issue (WE embedding missing).", errors.New("WE missing")
	}
	if bpeActualVocabSize <= 0 {
		return "Error: BPE tokenizer not properly initialized (vocab size 0).", errors.New("BPE vocab size 0")
	}


	g := NewGraph(false) // No backprop needed for inference
	var hiddenStates [][]*Mat // Start with nil hidden states for the first token

	// --- Prepare Input ---
	userToken := "[USER]"; botToken := "[BOT]"; eosToken := "[EOS]"
	userTokenID, hasUser := bpe.specialTokensMap[userToken]
	botTokenID, hasBot := bpe.specialTokensMap[botToken] // Used for trimming output
	eosTokenID, hasEOS := bpe.specialTokensMap[eosToken]
	unkTokenID, hasUnk := bpe.specialTokensMap["[UNK]"]

	// Construct prompt with special tokens
	// Prime with USER input and the BOT start token string
	promptText := fmt.Sprintf("%s %s %s", userToken, inputText, botToken)
	promptIDs := bpe.Encode(promptText) // Encode includes handling of unknowns based on BPE setup

	// Filter out only potentially problematic IDs (-1 if no UNK) before priming
	validPromptIDsForPriming := []int{}
	for _, id := range promptIDs {
		if id >= 0 && id < bpeActualVocabSize {
			validPromptIDsForPriming = append(validPromptIDsForPriming, id)
		} else {
			log.Printf("Warning: Invalid token ID %d in prompt, treating as UNK/skipping in priming.", id)
			if hasUnk {
				validPromptIDsForPriming = append(validPromptIDsForPriming, unkTokenID)
			} else {
				validPromptIDsForPriming = append(validPromptIDsForPriming, -1) // Let Lookup handle -1 as zero vector
			}
		}
	}


	if len(validPromptIDsForPriming) == 0 {
		// This could happen if input is empty and special tokens aren't in vocab? Unlikely.
		log.Println("Warning: No valid tokens found after encoding the prompt.")
		return "I couldn't process that input.", nil
	}


	// --- Prime the Hidden State ---
	// Process the prompt sequence token by token to set the initial hidden state
	currentTokenID := -1 // Keep track of the last valid token ID processed
	for _, tokenID := range validPromptIDsForPriming {
		// Lookup handles -1 returning zeros, forward pass proceeds but state change might be minimal
		// We need batch dimension 1 for inference
		x := g.Lookup(model["WE"], []int{tokenID}) // Shape [EmbeddingDim x 1]

		// Ensure hiddenStates are initialized correctly for the first token
		// ForwardMoEGRU handles initialization if hiddenStates is nil
		// Uses global hiddenSizes, numExperts
		forwardResult := ForwardMoEGRU(g, model, hiddenSizes, numExperts, x, hiddenStates)
		hiddenStates = forwardResult.H // Update hidden state for the next priming token

		// Only update currentTokenID if it was a valid token ID processed by Lookup
		if tokenID != -1 {
		    currentTokenID = tokenID
		}
	}


	if currentTokenID == -1 {
		// This happens if the prompt *only* contained invalid tokens (and no UNK token)
		log.Println("Error: Failed to set currentTokenId during priming (only invalid tokens?).")
		return "Error processing the input prompt.", errors.New("failed to set currentTokenId during priming")
	}

	// --- Generate Response Iteratively ---
	generatedResponseIDs := []int{}
	// Use maxLength parameter (which comes from flagMaxResponseLength in main)
	for t := 0; t < maxLength; t++ {
		if currentTokenID < 0 || currentTokenID >= bpeActualVocabSize {
			// Should not happen if priming finished with a valid token and sampling works
			log.Printf("Error: Invalid currentTokenId (%d) at start of generation step %d. Stopping.", currentTokenID, t)
			break
		}

		// Input for this step is the embedding of the previously generated/primed token
		x := g.Lookup(model["WE"], []int{currentTokenID}) // Input embedding [EmbeddingDim x 1]

		// Run the forward pass to get logits for the next token
		// Uses global hiddenSizes, numExperts
		forwardResult := ForwardMoEGRU(g, model, hiddenSizes, numExperts, x, hiddenStates)
		hiddenStates = forwardResult.H   // Update hidden state for the *next* generation step
		outputLogits := forwardResult.O // Shape [VocabSize x 1]

		// Sampling: Convert logits to probabilities and sample the next token
		probs := SoftmaxStandalone(outputLogits) // [VocabSize x 1]
		sample := rand.Float64()                 // Random number [0.0, 1.0)
		cumulativeProb := 0.0
		nextTokenID := -1 // Initialize to invalid

		// --- Sampling Logic with Sanity Checks ---
		probSum := 0.0
		validProbs := true
		// Check for NaNs/Infs and calculate sum for potential renormalization
		for i := 0; i < probs.N; i++ {
			probVal := probs.Get(i, 0)
			if math.IsNaN(probVal) || math.IsInf(probVal, 0) {
				// log.Printf("Warning: NaN/Inf probability at index %d in step %d. Setting to 0.", i, t)
				probs.Set(i, 0, 0.0) // Replace invalid probability with 0
				validProbs = false   // Mark that probabilities were invalid
			}
			probSum += probs.Get(i, 0)
		}

		// Renormalize if necessary (sum significantly different from 1 or contained invalids)
		if !validProbs || math.Abs(probSum-1.0) > 1e-5 {
			// log.Printf("Warning: Probs sum to %.6f in step %d. Renormalizing.", probSum, t)
			if probSum <= 1e-9 { // If sum is effectively zero, cannot renormalize
				log.Printf("Warning: Probability sum is near zero in step %d. Sampling uniformly.", t)
				nextTokenID = rand.Intn(bpeActualVocabSize) // Fallback: sample any token uniformly
				goto EndSampling // Skip standard sampling loop
			}
			// Renormalize valid probabilities
			renormFactor := 1.0 / probSum
			cumulativeProb = 0.0 // Reset cumulative probability for renormalized sampling
			for i := 0; i < probs.N; i++ {
				probs.Set(i, 0, probs.Get(i, 0)*renormFactor) // Scale probability
				cumulativeProb += probs.Get(i, 0)
				if sample < cumulativeProb && nextTokenID == -1 { // Find the first token whose cumulative prob exceeds sample
					nextTokenID = i
				}
			}
			// If due to floating point error sample >= final cumulativeProb (should be ~1.0), assign last token
			if nextTokenID == -1 {
				nextTokenID = bpeActualVocabSize - 1
			}
			goto EndSampling // Skip the standard sampling loop below as we've already sampled

		} else {
			// --- Standard Sampling (if probs were valid and sum close to 1) ---
			for i := 0; i < bpeActualVocabSize; i++ {
				cumulativeProb += probs.Get(i, 0)
				if sample < cumulativeProb {
					nextTokenID = i
					break // Found the sampled token
				}
			}
			// Handle potential edge case where sample == 1.0 (unlikely with float64) or rounding error
			if nextTokenID == -1 {
				nextTokenID = bpeActualVocabSize - 1
			}
		}

	EndSampling:
		// Check for stopping conditions AFTER sampling
		if (hasEOS && nextTokenID == eosTokenID) || (hasUser && nextTokenID == userTokenID) {
			// Stop if EOS or USER token is generated
			break
		}
		// Ensure sampled token is valid before adding and using for next step
		if nextTokenID < 0 || nextTokenID >= bpeActualVocabSize {
			log.Printf("Error: Sampled invalid token ID %d in step %d. Stopping generation.", nextTokenID, t)
			break // Stop generation if something went wrong with sampling
		}

		generatedResponseIDs = append(generatedResponseIDs, nextTokenID)
		currentTokenID = nextTokenID // Update the current token for the next iteration
	}

	// --- Decode and Post-process ---
	if len(generatedResponseIDs) == 0 {
		return "...", nil // Handle case where nothing was generated (e.g., first sampled token was EOS)
	}
	decodedString := bpe.Decode(generatedResponseIDs)

	// Remove potential leading [BOT] token if present (check using hasBot and botTokenID)
	if hasBot && len(generatedResponseIDs) > 0 && generatedResponseIDs[0] == botTokenID {
		botTokenString := ""
		// Check if botTokenID is valid before accessing vocab array
		if botTokenID >= 0 && botTokenID < len(bpe.vocabArray) {
			botTokenString = bpe.vocabArray[botTokenID]
			// Trim the decoded string if it starts with the BOT token representation
			if strings.HasPrefix(decodedString, botTokenString) {
				decodedString = strings.TrimPrefix(decodedString, botTokenString)
				// Also trim leading space that might remain after removing the token
                decodedString = strings.TrimPrefix(decodedString, " ")
			}
		}
	}


	// Final cleanup
	finalResponse := strings.TrimSpace(decodedString)
	if finalResponse == "" {
		finalResponse = "..." // Handle if only BOT token was generated and removed
	}
	return finalResponse, nil
}


//======================================================================
// --- Main Execution & Chat Interface ---
//======================================================================

func main() {
	rand.Seed(time.Now().UnixNano())
	log.SetFlags(log.LstdFlags | log.Lshortfile) // Add file/line number to logs

	// --- Define Flags ---
	// Training & Model Hyperparameters
	flag.IntVar(&flagBPEVocabSize, "bpe-vocab-size", 850, "Target vocabulary size for BPE training")
	flag.IntVar(&flagEmbeddingDimension, "embedding-dim", 96, "Dimension for token embeddings")
	flag.IntVar(&flagGRUHiddenSize, "gru-hidden-size", 96, "Hidden size for GRU layers (used if gru-layers > 0)")
	flag.IntVar(&flagGRULayers, "gru-layers", 2, "Number of GRU layers")
	flag.IntVar(&flagNumExperts, "num-experts", 6, "Number of experts in MoE layers")
	flag.IntVar(&flagTrainSeqLength, "seq-length", 80, "Sequence length for training")
	flag.IntVar(&flagBatchSize, "batch-size", 16, "Batch size for training")
	flag.IntVar(&flagEpochs, "epochs", 5, "Number of training epochs")
	flag.IntVar(&flagMaxResponseLength, "max-response", 260, "Maximum number of tokens to generate in response")

	// Optimizer Hyperparameters
	flag.Float64Var(&flagLearningRate, "lr", 0.001, "Learning rate for AdamW optimizer")
	flag.Float64Var(&flagWeightDecay, "wd", 0.01, "Weight decay for AdamW optimizer")
	flag.Float64Var(&flagEpsilonRMSNorm, "eps-rmsnorm", 1e-5, "Epsilon for RMSNorm stability")
	flag.Float64Var(&flagEpsilonAdamW, "eps-adamw", 1e-8, "Epsilon for AdamW optimizer stability")
	flag.Float64Var(&flagGradientClipValue, "grad-clip", 5.0, "Gradient clipping value")

	// Existing flags for paths and mode
	checkpointFlag := flag.String("checkpoint", "", "Path to checkpoint file to load and resume training/inference")
	bpeDataFlag := flag.String("bpe-data", "", "Path to the data file for BPE tokenizer training")
	modelDataFlag := flag.String("model-data", "", "Path to the data file for model training")
	trainFlag := flag.Bool("train", false, "Run/continue model training (requires -model-data)")

	// --- Parse Flags ---
	flag.Parse() // Parse all defined flags

	// --- Post-Flag Setup & Variable Assignment ---
	// Assign flag values to the existing global variables used by the core logic.
	// This avoids replacing EVERY instance of the old constants, only the key assignments.
	// These might be overridden if loading from a checkpoint.
	embeddingDimension = flagEmbeddingDimension
	hiddenSizes = make([]int, flagGRULayers) // Initialize based on flags
	for i := range hiddenSizes {
		hiddenSizes[i] = flagGRUHiddenSize
	}
	numExperts = flagNumExperts
	seqLength = flagTrainSeqLength
	batchSize = flagBatchSize
	// bpeActualVocabSize is set after BPE loading/training
	// Epochs, MaxResponseLength, LR etc. are used directly via their flag variables or solver state

	// Print effective configuration after parsing flags (before potential checkpoint load)
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
	startEpoch := 0
	var err error
	needsModelTraining := *trainFlag // Determine if model training is requested

	log.Println("Status: Initializing...")

	// Initialize BPE structure first
	bpe = NewBPE(BpeSpecialTokens)
	bpeIsReady := false // Track if BPE is loaded or trained

	// --- Loading or Initializing ---
	if *checkpointFlag != "" {
		// Attempt to load from checkpoint
		var loadedSolver *SolverAdamW
		var loadedBPE *BPE // Local var to receive loaded BPE
		startEpoch, model, loadedSolver, loadedBPE, err = loadCheckpoint(*checkpointFlag)
		if err != nil {
			log.Fatalf("FATAL: Failed to load checkpoint from %s: %v", *checkpointFlag, err)
		}
		solver = loadedSolver // Assign loaded solver to global variable
		bpe = loadedBPE       // Assign loaded BPE to global variable
		bpeIsReady = true     // BPE is ready from checkpoint

		// Global config vars like hiddenSizes, embeddingDimension, seqLength, batchSize etc.
		// are updated inside loadCheckpoint from the checkpoint's config.
		log.Println("--- Effective Configuration (After Checkpoint Load) ---")
		log.Printf("  BPEVocabSize (Target): %d", flagBPEVocabSize) // Show original target
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


		// Warn if BPE data was provided but ignored due to checkpoint
		if bpeDataPath != "" {
			log.Printf("Warning: -bpe-data ('%s') provided but ignored because a checkpoint was loaded.", bpeDataPath)
		}

		// Check if model data is needed for continued training
		if needsModelTraining && modelDataPath == "" {
			log.Fatal("FATAL: Training requested (-train) but no model data file specified (-model-data).")
		}

	} else {
		// No checkpoint - Initialize from scratch
		log.Println("No checkpoint specified. Initializing from scratch.")

		// Train BPE if BPE data is provided
		if bpeDataPath != "" {
			err = trainBPEFromFile(bpeDataPath) // Uses flagBPEVocabSize internally
			if err != nil {
				log.Fatalf("FATAL: Failed to train BPE tokenizer: %v", err)
			}
			bpeIsReady = true // BPE is ready after training
		} else {
			log.Println("Warning: No -bpe-data provided and no checkpoint. BPE tokenizer will be empty unless data is loaded later.")
			// BPE is not ready yet. Model data prep will fail if needed.
		}

		// Model training requires model data
		if needsModelTraining && modelDataPath == "" {
			log.Fatal("FATAL: Training requested (-train) but no model data file specified (-model-data).")
		}
	}

	// --- Prepare Model Data if needed for training ---
	if needsModelTraining {
		if !bpeIsReady {
			log.Fatal("FATAL: Cannot prepare model data because BPE tokenizer was not loaded or trained.")
		}
		if modelDataPath == "" {
			// This check should have been caught earlier, but double-check
			log.Fatal("FATAL: Model training requested (-train) but -model-data path is missing.")
		}

		log.Println("Preparing model data using the BPE tokenizer...")
		// Uses global seqLength, batchSize (set from flags or checkpoint)
		dataReady, dataErr := prepareModelData(modelDataPath)
		if dataErr != nil {
			log.Fatalf("FATAL: Model data preparation failed: %v", dataErr)
		}
		if !dataReady {
			log.Fatalf("FATAL: Model data preparation indicated failure.")
		}
		log.Println("Model data prepared successfully.")

		// Initialize Model and Optimizer *if* not loaded from checkpoint
		if *checkpointFlag == "" {
			if !bpeIsReady || bpeActualVocabSize == 0 {
				log.Fatal("FATAL: Cannot initialize model. BPE not ready or vocab size is zero.")
			}
			log.Println("Initializing new model and optimizer...")
			// hiddenSizes slice is already initialized based on flags
			// Use global embeddingDimension, hiddenSizes, bpeActualVocabSize, numExperts
			model = InitMoEGRU(bpeActualVocabSize, embeddingDimension, hiddenSizes, bpeActualVocabSize, numExperts)
			// Use flag variables for LR, Eps, WD
			solver = NewSolverAdamW(flagLearningRate, 0.9, 0.999, flagEpsilonAdamW, flagWeightDecay)

			// Calculate and Print Total Parameters for new model
			totalParams := 0
			if model != nil {
				keys := make([]string, 0, len(model))
				for k := range model { keys = append(keys, k) }
				sort.Strings(keys)
				for _, k := range keys {
					if m := model[k]; m != nil {
						totalParams += m.N * m.D
					}
				}
			}
			log.Printf("-------------------------------------")
			log.Printf("Total parameters for new model: %d", totalParams)
			log.Printf("-------------------------------------")
		}

		// --- Execute Training ---
		// Use flagEpochs as the target number of epochs
		if startEpoch < flagEpochs {
			log.Printf("Proceeding with model training from epoch %d up to target epoch %d...", startEpoch, flagEpochs)
			err = trainGRUModel(startEpoch) // Continue or start training
			if err != nil {
				log.Fatalf("FATAL: Model training failed: %v", err)
			}
			// trainingComplete is set inside trainGRUModel on success
		} else {
			log.Printf("Loaded checkpoint is already at or beyond the target epoch (%d >= %d). No further training needed.", startEpoch, flagEpochs)
			trainingComplete = true // Mark as complete based on checkpoint epoch
		}

	} else {
		// Not training the model
		if *checkpointFlag != "" {
			log.Println("Checkpoint loaded. Skipping model training as -train flag was not provided.")
			trainingComplete = true // Ready for chat based on loaded checkpoint
		} else {
			// No checkpoint and no training requested. What can we do?
			// If BPE was trained, we still don't have a model.
			// If BPE wasn't trained, we have nothing.
			log.Println("No checkpoint loaded and model training not requested (-train). Cannot proceed to chat.")
			trainingComplete = false // Not ready
		}
	}


	// --- Start Chat Interface ---
	if !trainingComplete {
		log.Fatal("FATAL: Model is not ready for chat. Ensure a checkpoint is loaded or training (-train with -bpe-data and -model-data) is completed.")
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

		if strings.ToLower(input) == "exit" || strings.ToLower(input) == "quit" {
			break
		}
		if input == "" { continue }

		// Indicate thinking process (optional)
		// fmt.Println("Bot: Thinking...")

		// Use flagMaxResponseLength for the generation limit
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
