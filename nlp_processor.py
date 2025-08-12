import re
import nltk
import spacy
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from collections import Counter
from textstat import flesch_reading_ease, flesch_kincaid_grade, automated_readability_index
import streamlit as st

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer

class NLPProcessor:
    """Natural Language Processing utilities for academic text analysis and enhancement"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Try to load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            st.warning("spaCy model 'en_core_web_sm' not found. Some features may be limited.")
            self.nlp = None
        
        # Academic vocabulary lists
        self.academic_vocabulary = self._load_academic_vocabulary()
        self.formal_replacements = self._load_formal_replacements()
        self.clarity_improvements = self._load_clarity_improvements()
    
    def _load_academic_vocabulary(self) -> set:
        """Load academic vocabulary word list"""
        return {
            'analyze', 'analysis', 'analytical', 'approach', 'area', 'assessment', 'assume',
            'authority', 'available', 'benefit', 'concept', 'conclude', 'conclusion', 'conduct',
            'consistent', 'constitute', 'context', 'contract', 'create', 'criteria', 'data',
            'definition', 'demonstrate', 'derive', 'design', 'distribution', 'economic',
            'environment', 'establish', 'estimate', 'evaluate', 'evidence', 'examine',
            'factor', 'feature', 'final', 'focus', 'formula', 'function', 'identify',
            'impact', 'implication', 'indicate', 'individual', 'initial', 'instance',
            'interpret', 'interpretation', 'investigate', 'involve', 'issue', 'journal',
            'label', 'legal', 'legislation', 'major', 'method', 'methodology', 'minor',
            'occur', 'option', 'outcome', 'overall', 'participate', 'percent', 'period',
            'policy', 'previous', 'primary', 'principle', 'procedure', 'process', 'project',
            'provide', 'publish', 'purpose', 'range', 'ratio', 'reason', 'receive',
            'reflect', 'region', 'relevant', 'remove', 'require', 'research', 'resource',
            'respond', 'response', 'result', 'role', 'section', 'sector', 'select',
            'significant', 'similar', 'source', 'specific', 'strategy', 'structure',
            'subsequent', 'summary', 'theory', 'traditional', 'transfer', 'trend',
            'undertake', 'unique', 'utilize', 'valid', 'variable', 'version', 'volume'
        }
    
    def _load_formal_replacements(self) -> Dict[str, str]:
        """Load formal language replacements"""
        return {
            # Contractions
            "don't": "do not",
            "can't": "cannot",
            "won't": "will not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "hasn't": "has not",
            "haven't": "have not",
            "hadn't": "had not",
            "doesn't": "does not",
            "didn't": "did not",
            "wouldn't": "would not",
            "shouldn't": "should not",
            "couldn't": "could not",
            "mustn't": "must not",
            
            # Informal expressions
            "a lot of": "numerous",
            "lots of": "many",
            "kind of": "somewhat",
            "sort of": "somewhat",
            "pretty good": "satisfactory",
            "really good": "excellent",
            "very good": "excellent",
            "okay": "acceptable",
            "ok": "acceptable",
            "big": "substantial",
            "huge": "substantial",
            "tiny": "minimal",
            "small": "limited",
            "get": "obtain",
            "got": "obtained",
            "show": "demonstrate",
            "find": "identify",
            "look at": "examine",
            "think about": "consider",
            "talk about": "discuss",
            "point out": "indicate",
            "bring up": "introduce",
            "come up": "emerge",
            "go up": "increase",
            "go down": "decrease",
            "figure out": "determine",
            "work out": "resolve"
        }
    
    def _load_clarity_improvements(self) -> Dict[str, str]:
        """Load clarity improvement patterns"""
        return {
            "in order to": "to",
            "due to the fact that": "because",
            "in spite of the fact that": "although",
            "it is important to note that": "notably",
            "it should be pointed out that": "notably",
            "it is interesting to note that": "interestingly",
            "with regard to": "regarding",
            "in relation to": "regarding",
            "in connection with": "regarding",
            "for the purpose of": "to",
            "for the reason that": "because",
            "in the event that": "if",
            "in the case of": "for",
            "under circumstances in which": "when",
            "prior to": "before",
            "subsequent to": "after",
            "in the vicinity of": "near",
            "in the course of": "during",
            "at the present time": "currently",
            "at this point in time": "now",
            "facilitate the process of": "enable",
            "make an assumption": "assume",
            "reach a conclusion": "conclude",
            "come to the conclusion": "conclude",
            "make a decision": "decide",
            "conduct an investigation": "investigate",
            "perform an analysis": "analyze",
            "carry out": "conduct"
        }
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Comprehensive text analysis"""
        
        analysis = {
            'basic_stats': self._get_basic_statistics(text),
            'readability': self._calculate_readability(text),
            'academic_features': self._analyze_academic_features(text),
            'sentence_analysis': self._analyze_sentences(text),
            'vocabulary_analysis': self._analyze_vocabulary(text),
            'structure_analysis': self._analyze_structure(text),
            'tone_analysis': self._analyze_tone(text)
        }
        
        return analysis
    
    def _get_basic_statistics(self, text: str) -> Dict[str, int]:
        """Calculate basic text statistics"""
        
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        
        # Filter out punctuation
        words_only = [word for word in words if word.isalnum()]
        
        return {
            'character_count': len(text),
            'word_count': len(words_only),
            'sentence_count': len(sentences),
            'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]),
            'avg_words_per_sentence': len(words_only) / max(1, len(sentences)),
            'avg_characters_per_word': sum(len(word) for word in words_only) / max(1, len(words_only))
        }
    
    def _calculate_readability(self, text: str) -> Dict[str, float]:
        """Calculate readability metrics"""
        
        try:
            return {
                'flesch_reading_ease': flesch_reading_ease(text),
                'flesch_kincaid_grade': flesch_kincaid_grade(text),
                'automated_readability_index': automated_readability_index(text)
            }
        except:
            return {
                'flesch_reading_ease': 0.0,
                'flesch_kincaid_grade': 0.0,
                'automated_readability_index': 0.0
            }
    
    def _analyze_academic_features(self, text: str) -> Dict[str, Any]:
        """Analyze academic writing features"""
        
        words = word_tokenize(text.lower())
        words_only = [word for word in words if word.isalnum()]
        
        # Academic vocabulary usage
        academic_words = [word for word in words_only if word in self.academic_vocabulary]
        academic_percentage = (len(academic_words) / max(1, len(words_only))) * 100
        
        # Passive voice detection
        passive_count = self._count_passive_voice(text)
        
        # Citation patterns
        citations = self._find_citations(text)
        
        # Hedging language
        hedging_words = ['may', 'might', 'could', 'would', 'should', 'possibly', 'perhaps', 
                        'probably', 'likely', 'unlikely', 'seems', 'appears', 'suggests']
        hedging_count = sum(text.lower().count(word) for word in hedging_words)
        
        return {
            'academic_vocabulary_percentage': academic_percentage,
            'passive_voice_count': passive_count,
            'citation_count': len(citations),
            'hedging_language_count': hedging_count,
            'nominalizations': self._count_nominalizations(text),
            'complex_sentences': self._count_complex_sentences(text)
        }
    
    def _analyze_sentences(self, text: str) -> Dict[str, Any]:
        """Analyze sentence-level features"""
        
        sentences = sent_tokenize(text)
        
        sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences]
        
        # Sentence type analysis
        declarative = sum(1 for s in sentences if s.strip().endswith('.'))
        interrogative = sum(1 for s in sentences if s.strip().endswith('?'))
        exclamatory = sum(1 for s in sentences if s.strip().endswith('!'))
        
        return {
            'average_sentence_length': np.mean(sentence_lengths) if sentence_lengths else 0,
            'sentence_length_variance': np.var(sentence_lengths) if sentence_lengths else 0,
            'longest_sentence': max(sentence_lengths) if sentence_lengths else 0,
            'shortest_sentence': min(sentence_lengths) if sentence_lengths else 0,
            'declarative_sentences': declarative,
            'interrogative_sentences': interrogative,
            'exclamatory_sentences': exclamatory,
            'sentence_starters': self._analyze_sentence_starters(sentences)
        }
    
    def _analyze_vocabulary(self, text: str) -> Dict[str, Any]:
        """Analyze vocabulary features"""
        
        words = word_tokenize(text.lower())
        words_only = [word for word in words if word.isalnum() and word not in self.stop_words]
        
        word_freq = Counter(words_only)
        unique_words = len(word_freq)
        total_words = len(words_only)
        
        # Lexical diversity (Type-Token Ratio)
        ttr = unique_words / max(1, total_words)
        
        # Word length analysis
        word_lengths = [len(word) for word in words_only]
        
        return {
            'unique_words': unique_words,
            'total_words': total_words,
            'type_token_ratio': ttr,
            'average_word_length': np.mean(word_lengths) if word_lengths else 0,
            'long_words_percentage': (sum(1 for length in word_lengths if length > 6) / max(1, len(word_lengths))) * 100,
            'most_common_words': word_freq.most_common(10)
        }
    
    def _analyze_structure(self, text: str) -> Dict[str, Any]:
        """Analyze document structure"""
        
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Transition words
        transition_words = [
            'however', 'furthermore', 'moreover', 'therefore', 'consequently', 'nevertheless',
            'nonetheless', 'additionally', 'similarly', 'conversely', 'in contrast',
            'on the other hand', 'for example', 'for instance', 'in particular',
            'specifically', 'indeed', 'in fact', 'notably', 'importantly'
        ]
        
        transition_count = sum(text.lower().count(word) for word in transition_words)
        
        # Discourse markers
        discourse_markers = [
            'first', 'second', 'third', 'finally', 'in conclusion', 'to summarize',
            'in summary', 'overall', 'in general', 'broadly speaking'
        ]
        
        discourse_count = sum(text.lower().count(marker) for marker in discourse_markers)
        
        return {
            'paragraph_count': len(paragraphs),
            'average_paragraph_length': np.mean([len(word_tokenize(p)) for p in paragraphs]) if paragraphs else 0,
            'transition_words_count': transition_count,
            'discourse_markers_count': discourse_count,
            'sections_identified': self._identify_sections(text)
        }
    
    def _analyze_tone(self, text: str) -> Dict[str, Any]:
        """Analyze writing tone and style"""
        
        # Formality indicators
        formal_words = ['utilize', 'demonstrate', 'constitute', 'facilitate', 'subsequently']
        informal_words = ['get', 'show', 'make', 'help', 'then']
        
        formal_count = sum(text.lower().count(word) for word in formal_words)
        informal_count = sum(text.lower().count(word) for word in informal_words)
        
        # Certainty vs uncertainty
        certainty_words = ['clearly', 'obviously', 'definitely', 'certainly', 'undoubtedly']
        uncertainty_words = ['perhaps', 'possibly', 'maybe', 'might', 'could']
        
        certainty_count = sum(text.lower().count(word) for word in certainty_words)
        uncertainty_count = sum(text.lower().count(word) for word in uncertainty_words)
        
        return {
            'formality_score': formal_count - informal_count,
            'certainty_score': certainty_count - uncertainty_count,
            'personal_pronouns': self._count_personal_pronouns(text),
            'emphatic_language': self._count_emphatic_language(text)
        }
    
    def enhance_academic_tone(self, text: str) -> str:
        """Enhance academic tone of the text"""
        
        enhanced_text = text
        
        # Apply formal replacements
        for informal, formal in self.formal_replacements.items():
            enhanced_text = re.sub(r'\b' + re.escape(informal) + r'\b', formal, enhanced_text, flags=re.IGNORECASE)
        
        # Apply clarity improvements
        for verbose, concise in self.clarity_improvements.items():
            enhanced_text = re.sub(re.escape(verbose), concise, enhanced_text, flags=re.IGNORECASE)
        
        # Strengthen weak language
        weak_strong = {
            r'\bI think\b': 'It is suggested that',
            r'\bWe believe\b': 'It is believed that',
            r'\bWe feel\b': 'It is considered that',
            r'\bIt seems like\b': 'It appears that',
            r'\bmight be\b': 'is likely to be',
            r'\bcould be\b': 'may be',
            r'\bprobably\b': 'likely',
            r'\bmaybe\b': 'potentially'
        }
        
        for weak, strong in weak_strong.items():
            enhanced_text = re.sub(weak, strong, enhanced_text, flags=re.IGNORECASE)
        
        return enhanced_text
    
    def improve_sentence_structure(self, text: str) -> str:
        """Improve sentence structure and flow"""
        
        sentences = sent_tokenize(text)
        improved_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Break very long sentences
            if len(word_tokenize(sentence)) > 35:
                # Try to split at conjunctions
                for conjunction in [', and', ', but', ', however,', ', therefore,']:
                    if conjunction in sentence:
                        parts = sentence.split(conjunction, 1)
                        if len(parts) == 2:
                            improved_sentences.append(parts[0].strip() + '.')
                            improved_sentences.append(parts[1].strip().capitalize())
                            break
                else:
                    improved_sentences.append(sentence)
            else:
                improved_sentences.append(sentence)
        
        return ' '.join(improved_sentences)
    
    def enhance_vocabulary(self, text: str) -> str:
        """Enhance vocabulary with more academic terms"""
        
        vocabulary_upgrades = {
            r'\bshow\b': 'demonstrate',
            r'\bget\b': 'obtain',
            r'\bfind\b': 'identify',
            r'\buse\b': 'utilize',
            r'\bhelp\b': 'facilitate',
            r'\bbig\b': 'substantial',
            r'\bsmall\b': 'minimal',
            r'\bstart\b': 'initiate',
            r'\bend\b': 'conclude',
            r'\bmake\b': 'create',
            r'\bgive\b': 'provide',
            r'\btell\b': 'indicate',
            r'\bsee\b': 'observe',
            r'\blook\b': 'examine',
            r'\bthink\b': 'consider',
            r'\bknow\b': 'understand',
            r'\bcome\b': 'emerge',
            r'\bgo\b': 'proceed',
            r'\bput\b': 'place'
        }
        
        enhanced_text = text
        for simple, academic in vocabulary_upgrades.items():
            enhanced_text = re.sub(simple, academic, enhanced_text, flags=re.IGNORECASE)
        
        return enhanced_text
    
    def improve_clarity(self, text: str) -> str:
        """Improve text clarity and readability"""
        
        # Remove redundancies
        redundancy_patterns = {
            r'\bthat\s+that\b': 'that',
            r'\bvery\s+very\b': 'extremely',
            r'\bmore\s+and\s+more\b': 'increasingly',
            r'\beach\s+and\s+every\b': 'all',
            r'\bfirst\s+and\s+foremost\b': 'primarily',
            r'\bunique\s+and\s+different\b': 'unique',
            r'\bbasic\s+fundamentals\b': 'fundamentals',
            r'\bfuture\s+plans\b': 'plans',
            r'\bpast\s+history\b': 'history',
            r'\bfinal\s+outcome\b': 'outcome'
        }
        
        clarified_text = text
        for redundant, clear in redundancy_patterns.items():
            clarified_text = re.sub(redundant, clear, clarified_text, flags=re.IGNORECASE)
        
        # Improve phrase clarity
        clarified_text = re.sub(r'\bin\s+the\s+amount\s+of\b', 'amounting to', clarified_text, flags=re.IGNORECASE)
        clarified_text = re.sub(r'\bin\s+the\s+field\s+of\b', 'in', clarified_text, flags=re.IGNORECASE)
        clarified_text = re.sub(r'\bin\s+the\s+area\s+of\b', 'in', clarified_text, flags=re.IGNORECASE)
        
        return clarified_text
    
    def fix_grammar_issues(self, text: str) -> str:
        """Fix common grammar issues"""
        
        # Common grammar fixes
        grammar_fixes = {
            r'\bwhich\s+are\b': 'that are',  # Often misused
            r'\bthe\s+reason\s+is\s+because\b': 'the reason is that',
            r'\bdifferent\s+than\b': 'different from',
            r'\bless\s+people\b': 'fewer people',
            r'\bamount\s+of\s+people\b': 'number of people',
            r'\bcompare\s+to\b': 'compared with',
            r'\bcenter\s+around\b': 'center on',
            r'\bbased\s+off\s+of\b': 'based on'
        }
        
        corrected_text = text
        for incorrect, correct in grammar_fixes.items():
            corrected_text = re.sub(incorrect, correct, corrected_text, flags=re.IGNORECASE)
        
        return corrected_text
    
    def _count_passive_voice(self, text: str) -> int:
        """Count instances of passive voice"""
        
        passive_patterns = [
            r'\b(?:is|are|was|were|been|being)\s+\w+ed\b',
            r'\b(?:is|are|was|were|been|being)\s+\w+en\b',
        ]
        
        count = 0
        for pattern in passive_patterns:
            count += len(re.findall(pattern, text, re.IGNORECASE))
        
        return count
    
    def _find_citations(self, text: str) -> List[str]:
        """Find citation patterns in text"""
        
        citation_patterns = [
            r'\([^)]*\d{4}[^)]*\)',  # (Author, 2023) style
            r'\[[^\]]*\d+[^\]]*\]',  # [1], [Author, 2023] style
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+\(\d{4}\)',  # Author (2023) style
        ]
        
        citations = []
        for pattern in citation_patterns:
            citations.extend(re.findall(pattern, text))
        
        return citations
    
    def _count_nominalizations(self, text: str) -> int:
        """Count nominalizations (noun forms of verbs)"""
        
        nominalization_suffixes = ['-tion', '-sion', '-ment', '-ance', '-ence', '-ity', '-ness']
        
        words = word_tokenize(text.lower())
        count = 0
        
        for word in words:
            if any(word.endswith(suffix.replace('-', '')) for suffix in nominalization_suffixes):
                count += 1
        
        return count
    
    def _count_complex_sentences(self, text: str) -> int:
        """Count complex sentences (containing subordinate clauses)"""
        
        subordinating_conjunctions = [
            'although', 'because', 'since', 'while', 'if', 'when', 'where',
            'whereas', 'unless', 'until', 'after', 'before', 'though'
        ]
        
        sentences = sent_tokenize(text)
        complex_count = 0
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(conj in sentence_lower for conj in subordinating_conjunctions):
                complex_count += 1
        
        return complex_count
    
    def _analyze_sentence_starters(self, sentences: List[str]) -> Dict[str, int]:
        """Analyze how sentences start"""
        
        starters = Counter()
        
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            if words:
                first_word = words[0]
                if first_word not in self.stop_words:
                    starters[first_word] += 1
        
        return dict(starters.most_common(10))
    
    def _identify_sections(self, text: str) -> List[str]:
        """Identify document sections"""
        
        section_patterns = [
            r'(?i)^(?:abstract|introduction|background|literature\s+review|methods?|methodology|results?|findings|discussion|conclusion|references?)\s*$'
        ]
        
        sections = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            for pattern in section_patterns:
                if re.match(pattern, line):
                    sections.append(line)
                    break
        
        return sections
    
    def _count_personal_pronouns(self, text: str) -> int:
        """Count personal pronouns"""
        
        personal_pronouns = ['i', 'we', 'you', 'me', 'us', 'my', 'our', 'your']
        
        words = word_tokenize(text.lower())
        return sum(1 for word in words if word in personal_pronouns)
    
    def _count_emphatic_language(self, text: str) -> int:
        """Count emphatic language"""
        
        emphatic_words = [
            'very', 'extremely', 'highly', 'greatly', 'significantly', 'remarkably',
            'notably', 'particularly', 'especially', 'indeed', 'certainly', 'clearly'
        ]
        
        words = word_tokenize(text.lower())
        return sum(1 for word in words if word in emphatic_words)
    
    def generate_writing_suggestions(self, text: str) -> List[str]:
        """Generate specific writing improvement suggestions"""
        
        analysis = self.analyze_text(text)
        suggestions = []
        
        # Readability suggestions
        if analysis['readability']['flesch_reading_ease'] < 30:
            suggestions.append("Consider simplifying complex sentences to improve readability.")
        
        # Sentence length suggestions
        if analysis['sentence_analysis']['average_sentence_length'] > 25:
            suggestions.append("Many sentences are quite long. Consider breaking them into shorter, clearer sentences.")
        
        # Academic vocabulary suggestions
        if analysis['academic_features']['academic_vocabulary_percentage'] < 10:
            suggestions.append("Consider using more academic vocabulary to enhance the scholarly tone.")
        
        # Passive voice suggestions
        if analysis['academic_features']['passive_voice_count'] > len(sent_tokenize(text)) * 0.3:
            suggestions.append("Consider reducing passive voice usage for more direct and engaging writing.")
        
        # Structure suggestions
        if analysis['structure_analysis']['transition_words_count'] < analysis['structure_analysis']['paragraph_count']:
            suggestions.append("Add more transition words to improve flow between paragraphs and ideas.")
        
        # Citation suggestions
        if analysis['academic_features']['citation_count'] == 0 and len(word_tokenize(text)) > 500:
            suggestions.append("Consider adding citations to support your arguments and claims.")
        
        # Vocabulary diversity suggestions
        if analysis['vocabulary_analysis']['type_token_ratio'] < 0.5:
            suggestions.append("Try to vary your vocabulary more to avoid repetition.")
        
        if not suggestions:
            suggestions.append("Your writing demonstrates good academic style. Continue refining based on specific requirements.")
        
        return suggestions
    
    def extract_key_terms(self, text: str, num_terms: int = 10) -> List[Tuple[str, int]]:
        """Extract key terms from text using frequency and academic vocabulary"""
        
        words = word_tokenize(text.lower())
        words_filtered = [
            word for word in words 
            if word.isalnum() and word not in self.stop_words and len(word) > 3
        ]
        
        # Weight academic vocabulary terms higher
        weighted_freq = Counter()
        for word in words_filtered:
            weight = 2 if word in self.academic_vocabulary else 1
            weighted_freq[word] += weight
        
        return weighted_freq.most_common(num_terms)
    
    def compare_texts(self, text1: str, text2: str) -> Dict[str, Any]:
        """Compare two texts and provide improvement metrics"""
        
        analysis1 = self.analyze_text(text1)
        analysis2 = self.analyze_text(text2)
        
        comparison = {
            'readability_improvement': {
                'flesch_score_change': analysis2['readability']['flesch_reading_ease'] - analysis1['readability']['flesch_reading_ease'],
                'grade_level_change': analysis2['readability']['flesch_kincaid_grade'] - analysis1['readability']['flesch_kincaid_grade']
            },
            'academic_improvement': {
                'vocabulary_change': analysis2['academic_features']['academic_vocabulary_percentage'] - analysis1['academic_features']['academic_vocabulary_percentage'],
                'passive_voice_change': analysis2['academic_features']['passive_voice_count'] - analysis1['academic_features']['passive_voice_count']
            },
            'structure_improvement': {
                'sentence_length_change': analysis2['sentence_analysis']['average_sentence_length'] - analysis1['sentence_analysis']['average_sentence_length'],
                'transition_words_change': analysis2['structure_analysis']['transition_words_count'] - analysis1['structure_analysis']['transition_words_count']
            }
        }
        
        return comparison
