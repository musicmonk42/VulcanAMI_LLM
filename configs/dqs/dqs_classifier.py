#!/usr/bin/env python3
"""
VulcanAMI Data Quality System - Main Scoring Engine
Comprehensive multi-dimensional data quality classification and scoring
"""

import json
import logging
import hashlib
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import redis
import psycopg2
from psycopg2.extras import RealDictCursor

# ML/NLP imports
try:
    import spacy
    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
    from sentence_transformers import SentenceTransformer
    HAS_ML = True
except ImportError:
    HAS_ML = False
    logging.warning("ML libraries not available. Some features will be disabled.")

# Graph analysis
try:
    import networkx as nx
    HAS_GRAPH = True
except ImportError:
    HAS_GRAPH = False
    logging.warning("NetworkX not available. Graph analysis disabled.")


@dataclass
class QualityScore:
    """Represents a complete quality score with all dimensions"""
    overall_score: float
    dimension_scores: Dict[str, float]
    category: str
    action: str
    labels: List[str] = field(default_factory=list)
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataQualityClassifier:
    """Main data quality classification engine"""
    
    def __init__(self, config_path: str = "/etc/dqs/classifier.json"):
        """Initialize the classifier with configuration"""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.redis_client = self._setup_redis()
        self.db_conn = self._setup_database()
        
        # Initialize ML models if available
        if HAS_ML and self.config['dimensions']['pii_detection']['enabled']:
            self._init_pii_models()
        
        if HAS_ML and self.config['dimensions']['semantic_validity']['enabled']:
            self._init_semantic_models()
        
        self.logger.info("DataQualityClassifier initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load classifier configuration"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger('DQS.Classifier')
    
    def _setup_redis(self) -> Optional[redis.Redis]:
        """Setup Redis connection for caching"""
        if not self.config['performance']['caching']['enabled']:
            return None
        
        try:
            cache_config = self.config['performance']['caching']
            return redis.Redis(
                host=cache_config['redis_host'],
                port=cache_config['redis_port'],
                db=cache_config['redis_db'],
                decode_responses=False
            )
        except Exception as e:
            self.logger.warning(f"Redis connection failed: {e}")
            return None
    
    def _setup_database(self) -> Optional[psycopg2.extensions.connection]:
        """Setup database connection"""
        try:
            return psycopg2.connect(
                host='postgres',
                port=5432,
                database='vulcanami',
                user='dqs',
                password='',  # Use environment variable
                cursor_factory=RealDictCursor
            )
        except Exception as e:
            self.logger.warning(f"Database connection failed: {e}")
            return None
    
    def _init_pii_models(self):
        """Initialize PII detection models"""
        self.logger.info("Initializing PII detection models...")
        
        pii_config = self.config['dimensions']['pii_detection']
        
        # Load transformer model for PII
        if pii_config['models']['transformer']['enabled']:
            model_name = pii_config['models']['transformer']['model']
            self.pii_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.pii_model = AutoModelForTokenClassification.from_pretrained(model_name)
            self.pii_pipeline = pipeline(
                "token-classification",
                model=self.pii_model,
                tokenizer=self.pii_tokenizer,
                aggregation_strategy="simple"
            )
        
        # Load spaCy NER model
        if pii_config['models']['spacy_ner']['enabled']:
            model_name = pii_config['models']['spacy_ner']['model']
            self.spacy_nlp = spacy.load(model_name)
        
        self.logger.info("PII models initialized")
    
    def _init_semantic_models(self):
        """Initialize semantic validation models"""
        self.logger.info("Initializing semantic models...")
        
        semantic_config = self.config['dimensions']['semantic_validity']
        
        if semantic_config['embedding_analysis']['enabled']:
            model_name = semantic_config['embedding_analysis']['model']
            self.embedding_model = SentenceTransformer(model_name)
        
        self.logger.info("Semantic models initialized")
    
    def classify(self, data: Any, data_type: str = "text", 
                 metadata: Optional[Dict] = None) -> QualityScore:
        """
        Main classification method - scores data across all dimensions
        
        Args:
            data: The data to classify
            data_type: Type of data (text, graph, structured, etc.)
            metadata: Additional metadata about the data
        
        Returns:
            QualityScore object with complete scoring information
        """
        start_time = time.time()
        metadata = metadata or {}
        
        # Check cache first
        cache_key = self._generate_cache_key(data, data_type, metadata)
        if self.redis_client and self.config['performance']['caching']['cache_scores']:
            cached_score = self._get_cached_score(cache_key)
            if cached_score:
                self.logger.debug(f"Returning cached score for {cache_key}")
                return cached_score
        
        # Calculate dimension scores
        dimension_scores = {}
        labels = []
        
        for dimension_name, dimension_config in self.config['weights'].items():
            if not dimension_config['enabled']:
                continue
            
            try:
                score, dimension_labels = self._score_dimension(
                    dimension_name, data, data_type, metadata
                )
                dimension_scores[dimension_name] = score
                labels.extend(dimension_labels)
                
                # Early exit for critical dimensions with very low scores
                if (dimension_config.get('critical') and score < 0.30 and
                    self.config['performance']['optimization']['early_termination']):
                    self.logger.info(f"Early termination: {dimension_name} score {score:.3f}")
                    break
                    
            except Exception as e:
                self.logger.error(f"Error scoring dimension {dimension_name}: {e}")
                dimension_scores[dimension_name] = 0.0
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(dimension_scores)
        
        # Determine category and action
        category, action = self._categorize_score(overall_score)
        
        # Create quality score object
        quality_score = QualityScore(
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            category=category,
            action=action,
            labels=list(set(labels)),
            metadata={
                'processing_time_ms': (time.time() - start_time) * 1000,
                'data_type': data_type,
                'classifier_version': self.config['version'],
                **metadata
            }
        )
        
        # Cache the result
        if self.redis_client:
            self._cache_score(cache_key, quality_score)
        
        # Log and audit
        self._audit_classification(quality_score, data, metadata)
        
        return quality_score
    
    def _score_dimension(self, dimension: str, data: Any, 
                        data_type: str, metadata: Dict) -> Tuple[float, List[str]]:
        """Score a specific quality dimension"""
        
        if dimension == "pii_confidence":
            return self._score_pii_confidence(data, data_type)
        elif dimension == "graph_completeness":
            return self._score_graph_completeness(data, metadata)
        elif dimension == "syntactic_completeness":
            return self._score_syntactic_completeness(data, data_type)
        elif dimension == "semantic_validity":
            return self._score_semantic_validity(data, data_type)
        elif dimension == "data_freshness":
            return self._score_data_freshness(data, metadata)
        elif dimension == "source_credibility":
            return self._score_source_credibility(data, metadata)
        elif dimension == "consistency_score":
            return self._score_consistency(data, metadata)
        elif dimension == "completeness_score":
            return self._score_completeness(data, data_type)
        else:
            self.logger.warning(f"Unknown dimension: {dimension}")
            return 0.5, []
    
    def _score_pii_confidence(self, data: Any, data_type: str) -> Tuple[float, List[str]]:
        """
        Score PII detection confidence
        Returns: (score, labels) where score is 1.0 for no PII, lower for detected PII
        """
        if not HAS_ML:
            return 1.0, []
        
        pii_config = self.config['dimensions']['pii_detection']
        if not pii_config['enabled']:
            return 1.0, []
        
        labels = []
        detections = []
        
        # Convert data to text
        text = self._extract_text(data, data_type)
        if not text:
            return 1.0, []
        
        # Run transformer-based PII detection
        if hasattr(self, 'pii_pipeline'):
            try:
                results = self.pii_pipeline(text[:512])  # Limit length
                for result in results:
                    if result['score'] > pii_config['confidence_threshold']:
                        detections.append({
                            'entity': result['entity_group'],
                            'score': result['score'],
                            'text': result['word']
                        })
                        labels.append('contains_pii')
            except Exception as e:
                self.logger.error(f"PII detection error: {e}")
        
        # Run spaCy NER
        if hasattr(self, 'spacy_nlp'):
            try:
                doc = self.spacy_nlp(text[:1000000])  # spaCy limit
                for ent in doc.ents:
                    if ent.label_ in pii_config['models']['spacy_ner']['entities']:
                        detections.append({
                            'entity': ent.label_,
                            'score': 1.0,
                            'text': ent.text
                        })
                        labels.append('contains_pii')
            except Exception as e:
                self.logger.error(f"spaCy NER error: {e}")
        
        # Calculate confidence score
        if not detections:
            return 1.0, []
        
        # More PII = lower score
        avg_confidence = np.mean([d['score'] for d in detections])
        num_detections = len(detections)
        
        # Penalize based on number and confidence of detections
        score = max(0.0, 1.0 - (num_detections * 0.1 * avg_confidence))
        
        return score, list(set(labels))
    
    def _score_graph_completeness(self, data: Any, metadata: Dict) -> Tuple[float, List[str]]:
        """Score graph completeness and connectivity"""
        if not HAS_GRAPH:
            return 0.5, []
        
        graph_config = self.config['dimensions']['graph_completeness']
        if not graph_config['enabled']:
            return 1.0, []
        
        labels = []
        
        # Extract or build graph from data
        if isinstance(data, nx.Graph):
            G = data
        else:
            G = self._extract_graph(data, metadata)
        
        if G is None or len(G.nodes()) == 0:
            return 0.0, ['incomplete_graph']
        
        metrics = graph_config['metrics']
        scores = []
        weights = []
        
        # Node coverage
        if metrics['node_coverage']['enabled']:
            expected_nodes = metadata.get('expected_nodes', len(G.nodes()))
            coverage = len(G.nodes()) / max(1, expected_nodes)
            scores.append(min(1.0, coverage))
            weights.append(metrics['node_coverage']['weight'])
            if coverage < metrics['node_coverage']['min_threshold']:
                labels.append('incomplete_graph')
        
        # Edge coverage
        if metrics['edge_coverage']['enabled']:
            expected_edges = metadata.get('expected_edges', len(G.edges()))
            coverage = len(G.edges()) / max(1, expected_edges)
            scores.append(min(1.0, coverage))
            weights.append(metrics['edge_coverage']['weight'])
        
        # Connectivity
        if metrics['connectivity']['enabled']:
            if nx.is_directed(G):
                connected = nx.is_weakly_connected(G)
            else:
                connected = nx.is_connected(G)
            scores.append(1.0 if connected else 0.5)
            weights.append(metrics['connectivity']['weight'])
            if not connected:
                labels.append('disconnected_graph')
        
        # Relationship density
        if metrics['relationship_density']['enabled']:
            n = len(G.nodes())
            if n > 1:
                possible_edges = n * (n - 1) / (2 if not nx.is_directed(G) else 1)
                density = len(G.edges()) / possible_edges
                scores.append(density)
                weights.append(metrics['relationship_density']['weight'])
        
        # Calculate weighted average
        if scores:
            final_score = np.average(scores, weights=weights)
        else:
            final_score = 0.5
        
        return final_score, list(set(labels))
    
    def _score_syntactic_completeness(self, data: Any, data_type: str) -> Tuple[float, List[str]]:
        """Score syntactic structure and format validity"""
        syntactic_config = self.config['dimensions']['syntactic_completeness']
        if not syntactic_config['enabled']:
            return 1.0, []
        
        labels = []
        scores = []
        
        # Parse validation
        try:
            if data_type == 'json':
                if isinstance(data, str):
                    json.loads(data)
                parse_score = 1.0
            elif data_type == 'xml':
                # XML validation
                parse_score = 1.0
            else:
                parse_score = 1.0
        except Exception as e:
            self.logger.error(f"Parse error: {e}")
            parse_score = 0.0
            labels.append('syntax_errors')
        
        scores.append(parse_score)
        
        # Field completeness
        if isinstance(data, dict):
            required_fields = syntactic_config['validation']['structure_validation'].get('required_fields', [])
            if required_fields:
                present = sum(1 for f in required_fields if f in data)
                field_score = present / len(required_fields)
                scores.append(field_score)
                if field_score < 1.0:
                    labels.append('incomplete_fields')
            else:
                scores.append(1.0)
        else:
            scores.append(1.0)
        
        final_score = np.mean(scores) if scores else 0.5
        return final_score, list(set(labels))
    
    def _score_semantic_validity(self, data: Any, data_type: str) -> Tuple[float, List[str]]:
        """Score semantic meaning and logical consistency"""
        if not HAS_ML:
            return 0.5, []
        
        semantic_config = self.config['dimensions']['semantic_validity']
        if not semantic_config['enabled']:
            return 1.0, []
        
        labels = []
        
        # Extract text for embedding
        text = self._extract_text(data, data_type)
        if not text or not hasattr(self, 'embedding_model'):
            return 0.5, []
        
        try:
            # Generate embedding
            embedding = self.embedding_model.encode(text[:512])
            
            # Compare to reference corpus if configured
            if semantic_config['embedding_analysis']['compare_to_corpus']:
                # Load or compute reference embeddings
                # This is simplified - in production, use vector DB
                similarity_score = 0.75  # Placeholder
            else:
                similarity_score = 0.75
            
            if similarity_score < semantic_config['embedding_analysis']['similarity_threshold']:
                labels.append('semantic_issues')
            
            return similarity_score, labels
            
        except Exception as e:
            self.logger.error(f"Semantic scoring error: {e}")
            return 0.5, []
    
    def _score_data_freshness(self, data: Any, metadata: Dict) -> Tuple[float, List[str]]:
        """Score data freshness and temporal relevance"""
        freshness_config = self.config['dimensions']['data_freshness']
        if not freshness_config['enabled']:
            return 1.0, []
        
        labels = []
        
        # Extract timestamp
        timestamp = None
        for field in freshness_config['timestamp_fields']:
            if isinstance(data, dict) and field in data:
                timestamp = data[field]
                break
            elif field in metadata:
                timestamp = metadata[field]
                break
        
        if not timestamp:
            return 0.5, []
        
        # Parse timestamp
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                return 0.5, []
        elif not isinstance(timestamp, datetime):
            return 0.5, []
        
        # Calculate age in days
        age_days = (datetime.utcnow() - timestamp).total_seconds() / 86400
        
        # Apply decay function
        temporal = freshness_config['temporal_analysis']
        half_life = temporal['half_life_days']
        
        if temporal['decay_function'] == 'exponential':
            score = max(temporal['min_score'], np.exp(-age_days / half_life))
        else:
            score = max(temporal['min_score'], 1.0 - (age_days / (2 * half_life)))
        
        if age_days > 180:
            labels.append('stale_data')
        
        return score, labels
    
    def _score_source_credibility(self, data: Any, metadata: Dict) -> Tuple[float, List[str]]:
        """Score source trustworthiness and authority"""
        credibility_config = self.config['dimensions']['source_credibility']
        if not credibility_config['enabled']:
            return 1.0, []
        
        labels = []
        source = metadata.get('source', '')
        
        if not source:
            return 0.5, []
        
        # Check whitelist
        whitelist = credibility_config['scoring']['whitelist']
        if whitelist['enabled']:
            for entry in whitelist['sources']:
                if self._match_pattern(source, entry['pattern']):
                    return entry['score'], []
        
        # Check blacklist
        blacklist = credibility_config['scoring']['blacklist']
        if blacklist['enabled']:
            for pattern in blacklist['sources']:
                if self._match_pattern(source, pattern):
                    labels.append('untrusted_source')
                    return 0.0, labels
        
        # Use default score
        return whitelist['default_score'], labels
    
    def _score_consistency(self, data: Any, metadata: Dict) -> Tuple[float, List[str]]:
        """Score data consistency"""
        consistency_config = self.config['dimensions']['consistency_score']
        if not consistency_config['enabled']:
            return 1.0, []
        
        labels = []
        scores = []
        
        # Internal consistency checks
        if isinstance(data, dict):
            # Check for duplicates, conflicts, etc.
            internal_score = 1.0  # Placeholder
            scores.append(internal_score)
        
        # Statistical consistency
        # Check for outliers, distribution anomalies
        statistical_score = 1.0  # Placeholder
        scores.append(statistical_score)
        
        if scores:
            labels.append('inconsistent_data') if np.mean(scores) < 0.6 else None
            return np.mean(scores), [l for l in labels if l]
        
        return 0.5, []
    
    def _score_completeness(self, data: Any, data_type: str) -> Tuple[float, List[str]]:
        """Score data completeness"""
        completeness_config = self.config['dimensions']['completeness_score']
        if not completeness_config['enabled']:
            return 1.0, []
        
        labels = []
        
        if isinstance(data, dict):
            total_fields = len(data)
            filled_fields = sum(1 for v in data.values() if v not in [None, '', 'N/A', 'null'])
            completeness = filled_fields / max(1, total_fields)
            
            if completeness < 0.7:
                labels.append('incomplete_fields')
            
            return completeness, labels
        
        return 1.0, []
    
    def _calculate_overall_score(self, dimension_scores: Dict[str, float]) -> float:
        """Calculate weighted overall quality score"""
        total_weight = 0.0
        weighted_sum = 0.0
        
        for dimension, score in dimension_scores.items():
            if dimension in self.config['weights']:
                weight = self.config['weights'][dimension]['weight']
                weighted_sum += score * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def _categorize_score(self, score: float) -> Tuple[str, str]:
        """Determine category and action based on score"""
        for category, config in self.config['classification']['categories'].items():
            if config['min_score'] <= score < config['max_score']:
                return category, config['action']
        
        return 'unacceptable', 'reject'
    
    def _extract_text(self, data: Any, data_type: str) -> str:
        """Extract text content from data"""
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            return json.dumps(data)
        else:
            return str(data)
    
    def _extract_graph(self, data: Any, metadata: Dict) -> Optional[nx.Graph]:
        """Extract or build graph from data"""
        if not HAS_GRAPH:
            return None
        
        # Placeholder - implement graph extraction logic
        return None
    
    def _match_pattern(self, text: str, pattern: str) -> bool:
        """Match text against pattern (supports wildcards)"""
        import re
        pattern = pattern.replace('.', '\\.').replace('*', '.*')
        return bool(re.match(pattern, text))
    
    def _generate_cache_key(self, data: Any, data_type: str, metadata: Dict) -> str:
        """Generate cache key for data"""
        content = f"{data_type}:{json.dumps(data, sort_keys=True)}:{json.dumps(metadata, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _get_cached_score(self, cache_key: str) -> Optional[QualityScore]:
        """Retrieve cached score"""
        try:
            cached = self.redis_client.get(f"dqs:score:{cache_key}")
            if cached:
                data = json.loads(cached)
                return QualityScore(**data)
        except Exception as e:
            self.logger.error(f"Cache retrieval error: {e}")
        return None
    
    def _cache_score(self, cache_key: str, score: QualityScore):
        """Cache quality score"""
        try:
            ttl = self.config['performance']['caching']['ttl']
            data = {
                'overall_score': score.overall_score,
                'dimension_scores': score.dimension_scores,
                'category': score.category,
                'action': score.action,
                'labels': score.labels,
                'confidence': score.confidence,
                'timestamp': score.timestamp.isoformat(),
                'metadata': score.metadata
            }
            self.redis_client.setex(
                f"dqs:score:{cache_key}",
                ttl,
                json.dumps(data)
            )
        except Exception as e:
            self.logger.error(f"Cache storage error: {e}")
    
    def _audit_classification(self, score: QualityScore, data: Any, metadata: Dict):
        """Audit classification result"""
        if not self.config['audit']['enabled']:
            return
        
        try:
            audit_record = {
                'timestamp': datetime.utcnow().isoformat(),
                'overall_score': score.overall_score,
                'category': score.category,
                'action': score.action,
                'dimension_scores': score.dimension_scores,
                'labels': score.labels,
                'data_hash': hashlib.sha256(str(data).encode()).hexdigest(),
                'metadata': metadata,
                'classifier_version': self.config['version']
            }
            
            # Write to audit log
            if self.config['audit']['log_all_classifications']:
                self.logger.info(f"AUDIT: {json.dumps(audit_record)}")
            
            # Store in database if available
            if self.db_conn:
                self._store_audit_record(audit_record)
                
        except Exception as e:
            self.logger.error(f"Audit error: {e}")
    
    def _store_audit_record(self, record: Dict):
        """Store audit record in database"""
        try:
            with self.db_conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO dqs.audit_log 
                    (timestamp, overall_score, category, action, dimension_scores, 
                     labels, data_hash, metadata, classifier_version)
                    VALUES (%(timestamp)s, %(overall_score)s, %(category)s, %(action)s,
                            %(dimension_scores)s::jsonb, %(labels)s, %(data_hash)s,
                            %(metadata)s::jsonb, %(classifier_version)s)
                """, record)
                self.db_conn.commit()
        except Exception as e:
            self.logger.error(f"Database audit error: {e}")
            self.db_conn.rollback()


def main():
    """Main entry point for testing"""
    classifier = DataQualityClassifier()
    
    # Test data
    test_data = {
        "name": "John Doe",
        "email": "john.doe@example.com",
        "ssn": "123-45-6789",
        "content": "This is a test document.",
        "created_at": "2025-11-14T00:00:00Z"
    }
    
    score = classifier.classify(test_data, data_type="json", metadata={"source": "test.com"})
    
    print(f"Overall Score: {score.overall_score:.3f}")
    print(f"Category: {score.category}")
    print(f"Action: {score.action}")
    print(f"Labels: {score.labels}")
    print(f"Dimension Scores:")
    for dim, s in score.dimension_scores.items():
        print(f"  {dim}: {s:.3f}")


if __name__ == "__main__":
    main()