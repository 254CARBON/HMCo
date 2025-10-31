"""
Content scanner for detecting PII and sensitive data using regex and ML heuristics.
"""
import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

logger = logging.getLogger(__name__)


class SensitivityLevel(Enum):
    """Data sensitivity classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class DataCategory(Enum):
    """Data category classifications"""
    PII = "pii"
    PHI = "phi"
    FINANCIAL = "financial"
    PROPRIETARY = "proprietary"
    TRADE_SECRET = "trade_secret"
    LOCATION = "location"
    NONE = "none"


@dataclass
class ColumnClassification:
    """Classification result for a column"""
    table_name: str
    column_name: str
    sensitivity_level: SensitivityLevel
    data_categories: List[DataCategory]
    confidence_score: float
    detected_patterns: List[str]
    sample_matches: List[str]
    recommended_policy: str
    reasoning: str


class ContentScanner:
    """Scans data content for PII and sensitive information"""
    
    def __init__(self):
        # Initialize Presidio analyzer for PII detection
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()
        
        self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
        
        # Regex patterns for common sensitive data
        self.patterns = {
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            'api_key': r'\b[A-Za-z0-9]{32,}\b',
            'account_number': r'\b\d{10,20}\b'
        }
    
    def scan_column(self, 
                    table_name: str,
                    column_name: str,
                    sample_data: List[str],
                    column_metadata: Optional[Dict[str, Any]] = None) -> ColumnClassification:
        """Scan a column for sensitive data"""
        
        # Convert to strings
        sample_data = [str(val) for val in sample_data if val is not None]
        
        if not sample_data:
            return self._create_empty_classification(table_name, column_name)
        
        # Run Presidio analysis
        presidio_results = self._analyze_with_presidio(sample_data)
        
        # Run regex pattern matching
        regex_results = self._analyze_with_regex(sample_data)
        
        # Combine results
        all_detected = presidio_results + regex_results
        
        # Determine sensitivity and categories
        sensitivity, categories = self._determine_sensitivity(all_detected, column_metadata)
        
        # Calculate confidence
        confidence = self._calculate_confidence(all_detected, sample_data)
        
        # Recommend policy
        policy = self._recommend_policy(sensitivity, categories)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(all_detected, sensitivity, categories)
        
        return ColumnClassification(
            table_name=table_name,
            column_name=column_name,
            sensitivity_level=sensitivity,
            data_categories=categories,
            confidence_score=confidence,
            detected_patterns=[d['entity_type'] for d in all_detected],
            sample_matches=[d['text'][:50] for d in all_detected[:5]],  # Truncate samples
            recommended_policy=policy,
            reasoning=reasoning
        )
    
    def _analyze_with_presidio(self, sample_data: List[str]) -> List[Dict[str, Any]]:
        """Analyze sample data with Presidio"""
        results = []
        
        for text in sample_data[:100]:  # Sample first 100 rows
            try:
                analysis = self.analyzer.analyze(
                    text=text,
                    language='en',
                    entities=None  # Detect all entity types
                )
                
                for result in analysis:
                    results.append({
                        'entity_type': result.entity_type,
                        'score': result.score,
                        'text': text[result.start:result.end]
                    })
            except Exception as e:
                logger.warning(f"Presidio analysis failed: {e}")
                continue
        
        return results
    
    def _analyze_with_regex(self, sample_data: List[str]) -> List[Dict[str, Any]]:
        """Analyze sample data with regex patterns"""
        results = []
        
        for text in sample_data[:100]:
            for pattern_name, pattern_regex in self.patterns.items():
                matches = re.findall(pattern_regex, text)
                for match in matches:
                    results.append({
                        'entity_type': pattern_name,
                        'score': 0.9,  # High confidence for regex matches
                        'text': match
                    })
        
        return results
    
    def _determine_sensitivity(self, 
                               detected: List[Dict[str, Any]],
                               metadata: Optional[Dict[str, Any]]) -> tuple:
        """Determine sensitivity level and data categories"""
        
        if not detected:
            return SensitivityLevel.PUBLIC, [DataCategory.NONE]
        
        # Map entity types to categories and sensitivities
        entity_map = {
            'PERSON': (SensitivityLevel.CONFIDENTIAL, DataCategory.PII),
            'EMAIL_ADDRESS': (SensitivityLevel.CONFIDENTIAL, DataCategory.PII),
            'PHONE_NUMBER': (SensitivityLevel.CONFIDENTIAL, DataCategory.PII),
            'ssn': (SensitivityLevel.RESTRICTED, DataCategory.PII),
            'credit_card': (SensitivityLevel.RESTRICTED, DataCategory.FINANCIAL),
            'US_SSN': (SensitivityLevel.RESTRICTED, DataCategory.PII),
            'CREDIT_CARD': (SensitivityLevel.RESTRICTED, DataCategory.FINANCIAL),
            'LOCATION': (SensitivityLevel.INTERNAL, DataCategory.LOCATION),
            'api_key': (SensitivityLevel.RESTRICTED, DataCategory.PROPRIETARY),
            'account_number': (SensitivityLevel.CONFIDENTIAL, DataCategory.FINANCIAL),
        }
        
        max_sensitivity = SensitivityLevel.PUBLIC
        categories = set()
        
        for detection in detected:
            entity_type = detection['entity_type']
            if entity_type in entity_map:
                sensitivity, category = entity_map[entity_type]
                
                # Take highest sensitivity
                if list(SensitivityLevel).index(sensitivity) > list(SensitivityLevel).index(max_sensitivity):
                    max_sensitivity = sensitivity
                
                categories.add(category)
        
        return max_sensitivity, list(categories) if categories else [DataCategory.NONE]
    
    def _calculate_confidence(self, detected: List[Dict[str, Any]], sample_data: List[str]) -> float:
        """Calculate confidence score"""
        if not sample_data:
            return 0.0
        
        if not detected:
            return 1.0  # High confidence it's NOT sensitive
        
        # Calculate percentage of samples with detections
        samples_with_detections = len(set(d['text'] for d in detected))
        total_samples = min(len(sample_data), 100)
        
        detection_rate = samples_with_detections / total_samples
        
        # Average score from detections
        avg_score = sum(d['score'] for d in detected) / len(detected)
        
        # Combine
        confidence = (detection_rate * 0.6) + (avg_score * 0.4)
        
        return min(confidence, 1.0)
    
    def _recommend_policy(self, sensitivity: SensitivityLevel, categories: List[DataCategory]) -> str:
        """Recommend data policy based on classification"""
        
        if sensitivity == SensitivityLevel.RESTRICTED:
            if DataCategory.PII in categories or DataCategory.FINANCIAL in categories:
                return "mask_all"  # Full masking
            else:
                return "deny_access"  # No access
        
        elif sensitivity == SensitivityLevel.CONFIDENTIAL:
            if DataCategory.PII in categories:
                return "mask_partial"  # Partial masking (e.g., last 4 digits)
            else:
                return "role_based_access"  # Restrict to specific roles
        
        elif sensitivity == SensitivityLevel.INTERNAL:
            return "authenticated_access"  # Require authentication
        
        else:  # PUBLIC
            return "open_access"
    
    def _generate_reasoning(self, 
                           detected: List[Dict[str, Any]],
                           sensitivity: SensitivityLevel,
                           categories: List[DataCategory]) -> str:
        """Generate human-readable reasoning"""
        
        if not detected:
            return "No sensitive data patterns detected. Column appears safe for open access."
        
        entity_types = list(set(d['entity_type'] for d in detected))
        category_names = ', '.join([c.value for c in categories])
        
        return f"Detected {len(detected)} instances of sensitive data ({', '.join(entity_types)}). " \
               f"Classified as {sensitivity.value} with categories: {category_names}. " \
               f"Recommended policy enforces appropriate access controls and masking."
    
    def _create_empty_classification(self, table_name: str, column_name: str) -> ColumnClassification:
        """Create classification for empty column"""
        return ColumnClassification(
            table_name=table_name,
            column_name=column_name,
            sensitivity_level=SensitivityLevel.PUBLIC,
            data_categories=[DataCategory.NONE],
            confidence_score=0.0,
            detected_patterns=[],
            sample_matches=[],
            recommended_policy="open_access",
            reasoning="No data available for classification"
        )


class DatasetScanner:
    """Scans entire datasets (tables) for sensitive data"""
    
    def __init__(self):
        self.column_scanner = ContentScanner()
    
    def scan_table(self, 
                   connection,
                   table_name: str,
                   sample_size: int = 1000) -> List[ColumnClassification]:
        """Scan all columns in a table"""
        
        logger.info(f"Scanning table: {table_name}")
        
        # Get column names
        columns_query = f"SELECT * FROM {table_name} LIMIT 0"
        cursor = connection.cursor()
        cursor.execute(columns_query)
        column_names = [desc[0] for desc in cursor.description]
        
        classifications = []
        
        for column_name in column_names:
            logger.info(f"  Scanning column: {column_name}")
            
            # Sample data
            sample_query = f"SELECT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL LIMIT {sample_size}"
            cursor.execute(sample_query)
            sample_data = [row[0] for row in cursor.fetchall()]
            
            # Classify column
            classification = self.column_scanner.scan_column(
                table_name=table_name,
                column_name=column_name,
                sample_data=sample_data
            )
            
            classifications.append(classification)
        
        return classifications
    
    def calculate_risk_score(self, classifications: List[ColumnClassification]) -> Dict[str, Any]:
        """Calculate overall risk score for dataset"""
        
        if not classifications:
            return {'risk_score': 0, 'risk_level': 'low'}
        
        # Weight by sensitivity
        weights = {
            SensitivityLevel.PUBLIC: 0,
            SensitivityLevel.INTERNAL: 1,
            SensitivityLevel.CONFIDENTIAL: 3,
            SensitivityLevel.RESTRICTED: 5
        }
        
        total_score = sum(
            weights[c.sensitivity_level] * c.confidence_score 
            for c in classifications
        )
        
        max_score = len(classifications) * 5  # Max if all RESTRICTED
        risk_score = (total_score / max_score) * 100 if max_score > 0 else 0
        
        if risk_score >= 75:
            risk_level = 'critical'
        elif risk_score >= 50:
            risk_level = 'high'
        elif risk_score >= 25:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'risk_score': round(risk_score, 2),
            'risk_level': risk_level,
            'total_columns': len(classifications),
            'sensitive_columns': len([c for c in classifications if c.sensitivity_level != SensitivityLevel.PUBLIC]),
            'restricted_columns': len([c for c in classifications if c.sensitivity_level == SensitivityLevel.RESTRICTED])
        }
