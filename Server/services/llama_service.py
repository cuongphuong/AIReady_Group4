"""
Llama Model Service
Load và sử dụng Llama model với GGUF format (llama-cpp-python)
"""
import os
import json
import logging
from typing import Optional, List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaService:
    def __init__(self, model_path: str = None):
        """Initialize Llama service
        
        Args:
            model_path: Path to GGUF model file
        """
        if model_path is None:
            # Auto-detect GGUF model
            base_dir = os.path.dirname(os.path.dirname(__file__))
            gguf_dir = os.path.join(base_dir, "gguf")
            
            # Find first .gguf file in directory
            if os.path.exists(gguf_dir):
                gguf_files = [f for f in os.listdir(gguf_dir) if f.endswith('.gguf')]
                if gguf_files:
                    model_path = os.path.join(gguf_dir, gguf_files[0])
                    logger.info(f"🔍 Found GGUF model: {gguf_files[0]}")
                else:
                    raise FileNotFoundError(f"No GGUF files found in {gguf_dir}")
            else:
                raise FileNotFoundError(f"GGUF directory not found: {gguf_dir}")
        
        self.model_path = model_path
        self.model = None
        self.loaded = False
        
        logger.info(f"Llama Service initialized (GGUF)")
        logger.info(f"Model path: {model_path}")
    
    def load_model(self):
        """Load model vào memory (lazy loading)"""
        if self.loaded and self.model is not None:
            logger.info("🔄 Model already loaded, reusing existing instance")
            return
        
        logger.info("\n" + "="*80)
        logger.info("🔄 Loading Llama GGUF model...")
        logger.info(f"📍 Model path: {self.model_path}")
        logger.info(f"💻 Running on CPU with llama-cpp-python")
        
        # Check if model file exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file does not exist: {self.model_path}")
        
        try:
            from llama_cpp import Llama
            
            logger.info("🧠 Loading GGUF model...")
            logger.info("⚡ Using 4-bit quantization (optimized for CPU)")
            
            # Load model with llama-cpp-python
            # n_ctx: context window size
            # n_threads: number of CPU threads (None = auto-detect)
            # n_gpu_layers: 0 for CPU only
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=2048,  # Context window
                n_threads=None,  # Auto-detect CPU threads
                n_gpu_layers=0,  # CPU only
                verbose=False
            )
            
            self.loaded = True
            logger.info(f"✅ Llama GGUF model loaded successfully on CPU")
            logger.info("="*80 + "\n")
        except ImportError:
            logger.error("❌ llama-cpp-python not installed")
            logger.error("💡 Install with: pip install llama-cpp-python")
            raise ImportError("llama-cpp-python is required for GGUF models")
        except Exception as e:
            logger.error(f"❌ Failed to load Llama model: {e}")
            logger.error(f"📍 Model path: {self.model_path}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def generate_text(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.1) -> str:
        """Generate text từ prompt"""
        logger.info("\n" + "-"*80)
        logger.info("🦙 GENERATE_TEXT")
        logger.info(f"📤 Prompt length: {len(prompt)} chars")
        logger.info(f"🔧 max_tokens: {max_new_tokens}, temperature: {temperature}")
        
        if not self.loaded:
            self.load_model()
        
        logger.info("🧠 Generating response with GGUF model...")
        
        # Generate with llama-cpp-python
        output = self.model(
            prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            stop=["<|eot_id|>", "<|end_of_text|>", "\n\n\n"],  # Llama 3 stop tokens
            echo=False  # Don't include prompt in output
        )
        
        generated_text = output['choices'][0]['text'].strip()
        
        logger.info("✅ Generation complete")
        logger.info(f"📥 Response length: {len(generated_text)} chars")
        logger.info(f"📝 Response preview: {generated_text[:200]}..." if len(generated_text) > 200 else f"📝 Response: {generated_text}")
        logger.info("-"*80 + "\n")
        
        return generated_text
    
    def classify_bug(self, description: str, labels: List[str], examples: List[Dict] = None) -> Dict:
        logger.info("\n" + "="*80)
        logger.info("🦙 LLAMA CLASSIFY_BUG")
        logger.info(f"📝 Input bug: {description[:100]}..." if len(description) > 100 else f"📝 Input bug: {description}")
        logger.info(f"🏷️ Available labels: {len(labels)}")
        
        # Ensure model is loaded (only loads once)
        if not self.loaded or self.model is None:
            self.load_model()
        
        # Build prompt
        labels_text = ", ".join(labels)
        logger.info(f"📚 Using {min(5, len(examples)) if examples else 0} examples")
        
        # Optimized prompt for Llama 3.1 Instruct
        # Use proper chat template format for better instruction following
        prompt = f"""<|start_header_id|>system<|end_header_id|>

You are an expert QA bug classifier. Analyze bug reports and classify them into the correct category. Always respond with valid JSON only.<|eot_id|><|start_header_id|>user<|end_header_id|>

Classify the following bug report into ONE category.

Available categories: {labels_text}

Bug report: "{description}"

Respond with ONLY a JSON object (no additional text):
{{
  "label": "the most appropriate category from the list",
  "reason": "concise explanation in Vietnamese (15-25 words)",
  "severity": "Low, Medium, High, or Critical"
}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{{"""
        
        logger.info(f"📤 Final prompt length: {len(prompt)} chars")

        try:
            response = self.generate_text(prompt, max_new_tokens=200, temperature=0.3)
            
            logger.info("🔍 Parsing Llama response...")
            
            # Clean response - remove any markdown or extra formatting
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()
            
            # If response doesn't start with {, add it (model might output without opening brace)
            if not response.startswith('{'):
                response = '{' + response
            
            # Extract first complete JSON object
            json_start = response.find('{')
            if json_start >= 0:
                # Find matching closing brace
                brace_count = 0
                json_end = json_start
                for i in range(json_start, len(response)):
                    if response[i] == '{':
                        brace_count += 1
                    elif response[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                if json_end > json_start:
                    json_str = response[json_start:json_end]
                    logger.info(f"📋 Found JSON: {json_str}")
                    result = json.loads(json_str)
                else:
                    logger.error("❌ Could not find complete JSON")
                    json_start = -1
            
            if json_start >= 0:
                
                # Validate label
                label = result.get('label', '')
                if label not in labels:
                    logger.warning(f"⚠️ Invalid label '{label}', trying to find closest match...")
                    # Try to find closest match
                    label_lower = label.lower()
                    for valid_label in labels:
                        if valid_label.lower() in label_lower or label_lower in valid_label.lower():
                            label = valid_label
                            logger.info(f"✅ Matched to valid label: {label}")
                            break
                else:
                    logger.info(f"✅ Valid label: {label}")
                
                final_result = {
                    'label': label if label in labels else labels[0],
                    'reason': result.get('reason', 'Classified by Llama model'),
                    'team': result.get('team'),
                    'severity': result.get('severity')
                }
                logger.info(f"✅ Classification result: {final_result}")
                logger.info("="*80 + "\n")
                return final_result
            
            # If we reach here, json_start was -1 or no valid JSON found
            if True:
                logger.error("❌ Could not find JSON in response")
                logger.info("🔄 Attempting fallback: extract label from text...")
                # Fallback: extract label from text
                for label in labels:
                    if label.lower() in response.lower():
                        fallback_result = {
                            'label': label,
                            'reason': 'Classified by keyword matching (Llama)',
                            'team': None,
                            'severity': None
                        }
                        logger.warning(f"⚠️ Fallback result: {fallback_result}")
                        logger.info("="*80 + "\n")
                        return fallback_result
                
                default_result = {
                    'label': labels[0],
                    'reason': 'Default classification (Llama)',
                    'team': None,
                    'severity': None
                }
                logger.warning(f"⚠️ Using default result: {default_result}")
                logger.info("="*80 + "\n")
                return default_result
        
        except Exception as e:
            logger.error(f"❌ Llama classification error: {e}")
            error_result = {
                'label': labels[0] if labels else 'Unknown',
                'reason': f'Error: {str(e)}',
                'team': None,
                'severity': None
            }
            logger.error(f"❌ Error result: {error_result}")
            logger.info("="*80 + "\n")
            return error_result
    
    def batch_classify_bugs(self, descriptions: List[str], labels: List[str], examples: List[Dict] = None) -> List[Dict]:
        """
        Batch classify multiple bugs in a single prompt (more efficient than individual calls)
        
        Args:
            descriptions: List of bug descriptions
            labels: Available classification labels
            examples: Few-shot examples (not used in batch mode)
        
        Returns:
            List of classification results
        """
        logger.info("\n" + "="*80)
        logger.info("🦙 LLAMA BATCH_CLASSIFY_BUGS")
        logger.info(f"📦 Batch size: {len(descriptions)}")
        logger.info(f"🏷️ Available labels: {len(labels)}")
        
        # Ensure model is loaded
        if not self.loaded or self.model is None:
            self.load_model()
        
        # Build batch prompt with all bugs
        labels_text = ", ".join(labels)
        
        # Create numbered list of bugs
        bugs_list = "\n".join([f"{i+1}. {desc}" for i, desc in enumerate(descriptions)])
        
        prompt = f"""<|start_header_id|>system<|end_header_id|>

You are an expert QA bug classifier. Analyze multiple bug reports and classify each into the correct category. Always respond with valid JSON array only.<|eot_id|><|start_header_id|>user<|end_header_id|>

Classify the following {len(descriptions)} bug reports into categories.

Available categories: {labels_text}

Bug reports:
{bugs_list}

Respond with ONLY a JSON array containing {len(descriptions)} objects (no additional text). Each object must have:
- "label": category from the list
- "reason": brief explanation in Vietnamese (15-25 words)
- "severity": Low, Medium, High, or Critical

JSON array:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

["""
        
        logger.info(f"📤 Batch prompt length: {len(prompt)} chars")
        
        try:
            # Generate with more tokens for batch
            max_tokens = min(300 + (len(descriptions) * 100), 2000)
            response = self.generate_text(prompt, max_new_tokens=max_tokens, temperature=0.3)
            
            logger.info("🔍 Parsing batch response...")
            
            # Clean response
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()
            
            # Add opening bracket if missing
            if not response.startswith('['):
                response = '[' + response
            
            # Find array boundaries
            array_start = response.find('[')
            if array_start >= 0:
                # Find matching closing bracket
                bracket_count = 0
                array_end = array_start
                for i in range(array_start, len(response)):
                    if response[i] == '[':
                        bracket_count += 1
                    elif response[i] == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            array_end = i + 1
                            break
                
                if array_end > array_start:
                    json_str = response[array_start:array_end]
                    logger.info(f"📋 Found JSON array: {len(json_str)} chars")
                    results_array = json.loads(json_str)
                    
                    if not isinstance(results_array, list):
                        raise ValueError("Response is not a JSON array")
                    
                    # Process and validate results
                    processed_results = []
                    for i, result in enumerate(results_array):
                        if i >= len(descriptions):
                            break
                        
                        label = result.get('label', '')
                        # Validate label
                        if label not in labels:
                            label_lower = label.lower()
                            for valid_label in labels:
                                if valid_label.lower() in label_lower or label_lower in valid_label.lower():
                                    label = valid_label
                                    break
                        
                        processed_results.append({
                            'label': label if label in labels else labels[0],
                            'reason': result.get('reason', 'Classified by Llama batch'),
                            'team': result.get('team'),
                            'severity': result.get('severity')
                        })
                    
                    # Fill missing results with defaults
                    while len(processed_results) < len(descriptions):
                        processed_results.append({
                            'label': labels[0],
                            'reason': 'Batch classification incomplete',
                            'team': None,
                            'severity': None
                        })
                    
                    logger.info(f"✅ Batch classification complete: {len(processed_results)} results")
                    logger.info("="*80 + "\n")
                    return processed_results[:len(descriptions)]
            
            # Fallback: classify individually if batch fails
            logger.warning("⚠️ Batch parsing failed, falling back to individual classification")
            logger.info("="*80 + "\n")
            return [self.classify_bug(desc, labels, examples) for desc in descriptions]
            
        except Exception as e:
            logger.error(f"❌ Batch classification error: {e}")
            logger.warning("⚠️ Falling back to individual classification")
            logger.info("="*80 + "\n")
            # Fallback to individual classification
            return [self.classify_bug(desc, labels, examples) for desc in descriptions]
    
    def unload_model(self):
        """Giải phóng memory"""
        if self.loaded:
            logger.info("🗑️ Unloading Llama model...")
            del self.model
            self.model = None
            self.loaded = False
            logger.info("✅ Llama model unloaded")
        else:
            logger.info("ℹ️ Model not loaded, nothing to unload")


# Global singleton instance
_llama_service = None
_llama_service_lock = False

def get_llama_service() -> LlamaService:
    """Get global Llama service instance (singleton pattern)"""
    global _llama_service, _llama_service_lock
    
    if _llama_service is None and not _llama_service_lock:
        _llama_service_lock = True
        try:
            logger.info("🔧 Initializing Llama service singleton...")
            _llama_service = LlamaService()
            # Pre-load model để tránh load lại mỗi request
            logger.info("⏳ Pre-loading model to avoid reload on each request...")
            _llama_service.load_model()
            logger.info("✅ Llama service singleton ready")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Llama service: {e}")
            _llama_service = None
            raise
        finally:
            _llama_service_lock = False
    
    return _llama_service
