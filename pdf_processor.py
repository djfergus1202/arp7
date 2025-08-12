import PyPDF2
import pdfplumber
import io
import re
import pandas as pd
from typing import List, Dict, Any, Optional
import streamlit as st
from datetime import datetime
import tempfile
import os

class PDFProcessor:
    """PDF processing utilities for extracting text, tables, and metadata"""
    
    def __init__(self):
        self.supported_formats = ['pdf']
    
    def extract_text(self, pdf_file) -> str:
        """Extract text content from PDF file"""
        
        try:
            # Reset file pointer
            pdf_file.seek(0)
            
            # Try pdfplumber first (better for complex layouts)
            text_content = self._extract_with_pdfplumber(pdf_file)
            
            if not text_content.strip():
                # Fallback to PyPDF2
                pdf_file.seek(0)
                text_content = self._extract_with_pypdf2(pdf_file)
            
            return text_content
            
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def _extract_with_pdfplumber(self, pdf_file) -> str:
        """Extract text using pdfplumber (better for tables and complex layouts)"""
        
        text_content = ""
        
        try:
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n\n"
        except Exception as e:
            st.warning(f"pdfplumber extraction failed: {str(e)}")
            return ""
        
        return text_content
    
    def _extract_with_pypdf2(self, pdf_file) -> str:
        """Extract text using PyPDF2 (fallback method)"""
        
        text_content = ""
        
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content += page_text + "\n\n"
                    
        except Exception as e:
            st.warning(f"PyPDF2 extraction failed: {str(e)}")
            return ""
        
        return text_content
    
    def extract_tables(self, pdf_file) -> List[pd.DataFrame]:
        """Extract tables from PDF using pdfplumber"""
        
        tables = []
        
        try:
            pdf_file.seek(0)
            
            with pdfplumber.open(pdf_file) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    
                    for table_num, table in enumerate(page_tables):
                        if table and len(table) > 1:  # At least header + 1 row
                            # Convert to DataFrame
                            df = pd.DataFrame(table[1:], columns=table[0])
                            
                            # Clean empty rows and columns
                            df = df.dropna(how='all').dropna(axis=1, how='all')
                            
                            # Add metadata
                            df.attrs['page'] = page_num + 1
                            df.attrs['table_number'] = table_num + 1
                            
                            tables.append(df)
            
        except Exception as e:
            st.warning(f"Error extracting tables: {str(e)}")
        
        return tables
    
    def extract_metadata(self, pdf_file) -> Dict[str, Any]:
        """Extract PDF metadata"""
        
        metadata = {}
        
        try:
            pdf_file.seek(0)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Basic metadata
            if pdf_reader.metadata:
                metadata.update({
                    'title': pdf_reader.metadata.get('/Title', 'Unknown'),
                    'author': pdf_reader.metadata.get('/Author', 'Unknown'),
                    'subject': pdf_reader.metadata.get('/Subject', 'Unknown'),
                    'creator': pdf_reader.metadata.get('/Creator', 'Unknown'),
                    'producer': pdf_reader.metadata.get('/Producer', 'Unknown'),
                    'creation_date': pdf_reader.metadata.get('/CreationDate', 'Unknown'),
                    'modification_date': pdf_reader.metadata.get('/ModDate', 'Unknown'),
                })
            
            # Document statistics
            metadata.update({
                'num_pages': len(pdf_reader.pages),
                'file_size': pdf_file.size if hasattr(pdf_file, 'size') else 'Unknown',
                'encrypted': pdf_reader.is_encrypted,
            })
            
        except Exception as e:
            st.warning(f"Error extracting metadata: {str(e)}")
        
        return metadata
    
    def get_page_count(self, pdf_file) -> int:
        """Get number of pages in PDF"""
        
        try:
            pdf_file.seek(0)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            return len(pdf_reader.pages)
        except Exception as e:
            st.warning(f"Error getting page count: {str(e)}")
            return 0
    
    def extract_study_data(self, pdf_file) -> Dict[str, Any]:
        """Extract structured study data from research papers"""
        
        text_content = self.extract_text(pdf_file)
        study_data = {}
        
        try:
            # Extract title
            study_data['title'] = self._extract_title(text_content)
            
            # Extract authors
            study_data['authors'] = self._extract_authors(text_content)
            
            # Extract publication year
            study_data['year'] = self._extract_year(text_content)
            
            # Extract abstract
            study_data['abstract'] = self._extract_abstract(text_content)
            
            # Extract study characteristics
            study_data['sample_size'] = self._extract_sample_size(text_content)
            study_data['study_design'] = self._extract_study_design(text_content)
            
            # Extract statistical results
            study_data['effect_sizes'] = self._extract_effect_sizes(text_content)
            study_data['p_values'] = self._extract_p_values(text_content)
            study_data['confidence_intervals'] = self._extract_confidence_intervals(text_content)
            
            # Extract references
            study_data['references'] = self._extract_references(text_content)
            
        except Exception as e:
            st.warning(f"Error extracting study data: {str(e)}")
        
        return study_data
    
    def _extract_title(self, text: str) -> str:
        """Extract paper title"""
        lines = text.split('\n')
        
        # Look for title in first few lines
        for i, line in enumerate(lines[:10]):
            line = line.strip()
            if len(line) > 10 and not line.isupper() and not line.isdigit():
                # Check if this looks like a title
                if any(word in line.lower() for word in ['study', 'analysis', 'effect', 'association', 'trial']):
                    return line
        
        # Fallback: return first non-empty line
        for line in lines:
            if line.strip() and len(line.strip()) > 10:
                return line.strip()
        
        return "Title not found"
    
    def _extract_authors(self, text: str) -> List[str]:
        """Extract author names"""
        authors = []
        
        # Common author patterns
        author_patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*),?\s*([A-Z]\.?\s*)+',  # LastName, F.M.
            r'([A-Z]\.?\s*)+\s+([A-Z][a-z]+)',  # F.M. LastName
        ]
        
        lines = text.split('\n')[:20]  # Check first 20 lines
        
        for line in lines:
            for pattern in author_patterns:
                matches = re.findall(pattern, line)
                for match in matches:
                    if isinstance(match, tuple):
                        author = ' '.join(match).strip()
                    else:
                        author = match.strip()
                    
                    if len(author) > 3 and author not in authors:
                        authors.append(author)
        
        return authors[:10]  # Limit to first 10 authors
    
    def _extract_year(self, text: str) -> Optional[int]:
        """Extract publication year"""
        
        # Look for 4-digit years in first part of text
        year_pattern = r'\b(19|20)\d{2}\b'
        matches = re.findall(year_pattern, text[:2000])
        
        if matches:
            years = [int(match) for match in matches if 1950 <= int(match) <= datetime.now().year]
            if years:
                return max(years)  # Return most recent year found
        
        return None
    
    def _extract_abstract(self, text: str) -> str:
        """Extract abstract section"""
        
        # Look for abstract section
        abstract_pattern = r'(?i)abstract[\s\n]+(.*?)(?=\n\s*(?:introduction|keywords|1\.|background))'
        match = re.search(abstract_pattern, text, re.DOTALL)
        
        if match:
            abstract = match.group(1).strip()
            # Clean up the abstract
            abstract = re.sub(r'\s+', ' ', abstract)
            return abstract[:1000]  # Limit length
        
        return "Abstract not found"
    
    def _extract_sample_size(self, text: str) -> List[int]:
        """Extract sample sizes from text"""
        
        sample_sizes = []
        
        # Common patterns for sample sizes
        patterns = [
            r'[Nn]\s*=\s*(\d+)',
            r'sample\s+size[^0-9]*(\d+)',
            r'(\d+)\s+participants',
            r'(\d+)\s+subjects',
            r'(\d+)\s+patients',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                n = int(match)
                if 10 <= n <= 100000:  # Reasonable sample size range
                    sample_sizes.append(n)
        
        return list(set(sample_sizes))  # Remove duplicates
    
    def _extract_study_design(self, text: str) -> List[str]:
        """Extract study design information"""
        
        designs = []
        
        design_patterns = [
            r'randomized\s+controlled\s+trial',
            r'cohort\s+study',
            r'case[_\s]control\s+study',
            r'cross[_\s]sectional\s+study',
            r'systematic\s+review',
            r'meta[_\s]analysis',
            r'observational\s+study',
            r'experimental\s+study',
        ]
        
        for pattern in design_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                design = re.findall(pattern, text, re.IGNORECASE)[0]
                designs.append(design.replace('_', '-'))
        
        return designs
    
    def _extract_effect_sizes(self, text: str) -> List[float]:
        """Extract effect sizes from text"""
        
        effect_sizes = []
        
        # Common effect size patterns
        patterns = [
            r'(?:cohen\'?s?\s+)?d\s*=\s*([-+]?\d*\.?\d+)',
            r'hedges\'?s?\s+g\s*=\s*([-+]?\d*\.?\d+)',
            r'effect\s+size[^0-9]*([-+]?\d*\.?\d+)',
            r'OR\s*=\s*(\d*\.?\d+)',
            r'odds\s+ratio[^0-9]*(\d*\.?\d+)',
            r'r\s*=\s*([-+]?0\.\d+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    es = float(match)
                    if -10 <= es <= 10:  # Reasonable effect size range
                        effect_sizes.append(es)
                except ValueError:
                    continue
        
        return effect_sizes
    
    def _extract_p_values(self, text: str) -> List[float]:
        """Extract p-values from text"""
        
        p_values = []
        
        # P-value patterns
        patterns = [
            r'p\s*[<>=]\s*(0?\.\d+)',
            r'p[_\s]*value[^0-9]*(0?\.\d+)',
            r'significance[^0-9]*(0?\.\d+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    p = float(match)
                    if 0 <= p <= 1:
                        p_values.append(p)
                except ValueError:
                    continue
        
        return p_values
    
    def _extract_confidence_intervals(self, text: str) -> List[tuple]:
        """Extract confidence intervals from text"""
        
        confidence_intervals = []
        
        # CI patterns
        patterns = [
            r'(?:95%\s*)?CI[^0-9]*([-+]?\d*\.?\d+)[^0-9]+([-+]?\d*\.?\d+)',
            r'\[([-+]?\d*\.?\d+)[,\s]+([-+]?\d*\.?\d+)\]',
            r'confidence\s+interval[^0-9]*([-+]?\d*\.?\d+)[^0-9]+([-+]?\d*\.?\d+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    lower = float(match[0])
                    upper = float(match[1])
                    if lower < upper and -100 <= lower <= 100 and -100 <= upper <= 100:
                        confidence_intervals.append((lower, upper))
                except (ValueError, IndexError):
                    continue
        
        return confidence_intervals
    
    def _extract_references(self, text: str) -> List[str]:
        """Extract reference list from text"""
        
        references = []
        
        # Look for references section
        ref_section_pattern = r'(?i)references?\s*\n(.*?)(?=\n\s*(?:appendix|supplementary|figure|table|\Z))'
        match = re.search(ref_section_pattern, text, re.DOTALL)
        
        if match:
            ref_text = match.group(1)
            
            # Split references by common patterns
            ref_patterns = [
                r'\n\d+\.\s+',  # Numbered references
                r'\n[A-Z][a-z]+,\s+[A-Z]\.',  # Author, X. format
            ]
            
            for pattern in ref_patterns:
                refs = re.split(pattern, ref_text)
                references.extend([ref.strip() for ref in refs if len(ref.strip()) > 50])
        
        return references[:50]  # Limit to 50 references
    
    def search_text(self, pdf_file, search_terms: List[str]) -> Dict[str, List[str]]:
        """Search for specific terms in PDF and return contexts"""
        
        text_content = self.extract_text(pdf_file)
        results = {}
        
        for term in search_terms:
            matches = []
            
            # Case-insensitive search
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            
            for match in pattern.finditer(text_content):
                start = max(0, match.start() - 100)
                end = min(len(text_content), match.end() + 100)
                context = text_content[start:end].strip()
                matches.append(context)
            
            results[term] = matches
        
        return results
    
    def extract_figures_captions(self, pdf_file) -> List[str]:
        """Extract figure captions from PDF"""
        
        text_content = self.extract_text(pdf_file)
        captions = []
        
        # Figure caption patterns
        patterns = [
            r'(?i)figure\s+\d+[.:]\s*([^\n]+(?:\n[^\n]*)*?)(?=\n\s*(?:figure|table|references?|\Z))',
            r'(?i)fig\.\s*\d+[.:]\s*([^\n]+(?:\n[^\n]*)*?)(?=\n\s*(?:fig\.|table|references?|\Z))',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text_content, re.MULTILINE)
            captions.extend([match.strip() for match in matches])
        
        return captions
    
    def validate_pdf(self, pdf_file) -> Dict[str, Any]:
        """Validate PDF file and return quality metrics"""
        
        validation_results = {
            'is_valid': False,
            'is_readable': False,
            'has_text': False,
            'has_tables': False,
            'page_count': 0,
            'text_length': 0,
            'issues': []
        }
        
        try:
            # Basic validation
            pdf_file.seek(0)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            validation_results['is_valid'] = True
            validation_results['page_count'] = len(pdf_reader.pages)
            
            if pdf_reader.is_encrypted:
                validation_results['issues'].append("PDF is encrypted")
                return validation_results
            
            # Text extraction test
            text_content = self.extract_text(pdf_file)
            validation_results['text_length'] = len(text_content)
            validation_results['has_text'] = len(text_content.strip()) > 100
            validation_results['is_readable'] = True
            
            # Table detection
            tables = self.extract_tables(pdf_file)
            validation_results['has_tables'] = len(tables) > 0
            
            # Quality checks
            if validation_results['text_length'] < 500:
                validation_results['issues'].append("Very short text content - may be image-based PDF")
            
            if not validation_results['has_text']:
                validation_results['issues'].append("No readable text found - may require OCR")
            
        except Exception as e:
            validation_results['issues'].append(f"PDF processing error: {str(e)}")
        
        return validation_results
