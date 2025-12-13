import os
import json
import logging
import asyncio
from typing import List, Dict, Optional, Any, Literal
from datetime import datetime
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.language_models.base import BaseLanguageModel
from pydantic import BaseModel, Field
from pinecone import Pinecone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "bugrecordindex")
PINECONE_HOST = os.getenv("PINECONE_HOST")

DB_OPENAI_API_KEY = os.getenv("DB_OPENAI_API_KEY")
DB_OPENAI_API_BASE = os.getenv("DB_OPENAI_API_BASE_URL")
DB_MODEL_NAME = os.getenv("DB_MODEL_NAME", "text-embedding-3-small")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE_URL")
OPENAI_MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5")


# Pydantic models for structured output
class BugClassification(BaseModel):
    """Schema for bug classification result"""
    label: str = Field(description="Classification label from available categories")
    reason: str = Field(description="Concise explanation in Vietnamese (< 30 words)")
    team: Optional[str] = Field(default=None, description="Team responsible")
    severity: Optional[str] = Field(default=None, description="Bug severity: Low, Medium, High, Critical")


class BatchBugClassification(BaseModel):
    """Schema for batch classification result"""
    index: int = Field(description="Bug index in the batch")
    label: str = Field(description="Classification label")
    reason: str = Field(description="Explanation in Vietnamese")
    team: Optional[str] = Field(default=None, description="Team responsible")
    severity: Optional[str] = Field(default=None, description="Bug severity")


class BatchClassificationResult(BaseModel):
    """Container for batch results"""
    classifications: List[BatchBugClassification]


class LlamaProvider:
    """Wrapper for Llama model to work with LangChain interface"""
    
    def __init__(self, model_path: str = None):
        """Initialize Llama provider
        
        Args:
            model_path: Path to GGUF model file
        """
        if model_path is None:
            # Auto-detect GGUF model
            base_dir = os.path.dirname(os.path.dirname(__file__))
            gguf_dir = os.path.join(base_dir, "gguf")
            
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
    
    def load_model(self):
        """Load Llama model (lazy loading)"""
        if self.loaded and self.model is not None:
            return
        
        logger.info(f"🦙 Loading Llama GGUF model from {self.model_path}")
        
        try:
            from llama_cpp import Llama
            
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=2048,
                n_threads=6,
                n_batch=512,
                n_gpu_layers=0,
                use_mlock=True,
                use_mmap=True,
                verbose=False
            )
            
            self.loaded = True
            logger.info("✅ Llama model loaded successfully")
        except ImportError:
            raise ImportError("llama-cpp-python is required. Install with: pip install llama-cpp-python")
        except Exception as e:
            logger.error(f"❌ Failed to load Llama model: {e}")
            raise
    
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        """Generate text from prompt"""
        if not self.loaded:
            self.load_model()
        
        output = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            top_k=40,
            repeat_penalty=1.1,
            stop=["<|eot_id|>", "<|end_of_text|>", "\n\n\n"],
            echo=False
        )
        
        return output['choices'][0]['text'].strip()
    
    async def agenerate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        """Async wrapper for generate"""
        return await asyncio.to_thread(self.generate, prompt, max_tokens, temperature)


class UnifiedLLMService:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.gpt_llm = None
        self.llama_provider = None
        
        logger.info("🚀 UnifiedLLMService initialized")
    
    # ========================
    # Embeddings & Vector Store
    # ========================
    
    def get_embeddings(self) -> OpenAIEmbeddings:
        """Get OpenAI embeddings instance (768 dimensions for Pinecone)"""
        if self.embeddings is None:
            logger.info(f"Creating embeddings: {DB_MODEL_NAME} (768 dims)")
            self.embeddings = OpenAIEmbeddings(
                model=DB_MODEL_NAME,
                openai_api_key=DB_OPENAI_API_KEY,
                openai_api_base=DB_OPENAI_API_BASE,
                dimensions=768,
            )
        return self.embeddings
    
    def get_vectorstore(self) -> PineconeVectorStore:
        """Get Pinecone vector store with LangChain wrapper"""
        if self.vectorstore is None:
            logger.info(f"Connecting to Pinecone: {PINECONE_INDEX_NAME}")
            
            pc = Pinecone(api_key=PINECONE_API_KEY)
            embeddings = self.get_embeddings()
            
            if PINECONE_HOST:
                index = pc.Index(PINECONE_INDEX_NAME, host=PINECONE_HOST)
            else:
                index = pc.Index(PINECONE_INDEX_NAME)
            
            self.vectorstore = PineconeVectorStore(
                index=index,
                embedding=embeddings,
                text_key="text",
                namespace=""
            )
            
            logger.info("✅ Pinecone VectorStore ready")
        
        return self.vectorstore
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform semantic similarity search in vector store"""
        embeddings = self.get_embeddings()
        query_vector = embeddings.embed_query(query)
        
        pc = Pinecone(api_key=PINECONE_API_KEY)
        if PINECONE_HOST:
            index = pc.Index(PINECONE_INDEX_NAME, host=PINECONE_HOST)
        else:
            index = pc.Index(PINECONE_INDEX_NAME)
        
        results = index.query(
            vector=query_vector,
            top_k=k,
            include_metadata=True,
            filter=filter
        )
        
        formatted_results = []
        for match in results.matches:
            meta = match.metadata if match.metadata else {}
            similarity = float(match.score) if match.score else 0.0
            
            formatted_results.append({
                'id': match.id,
                'text': meta.get('text', ''),
                'metadata': meta,
                'score': match.score,
                'similarity': similarity
            })
        
        return formatted_results
    
    def add_bug_to_vectorstore(
        self,
        bug_text: str,
        label: str,
        reason: str,
        team: Optional[str] = None,
        severity: Optional[str] = None,
        bug_id: Optional[str] = None
    ) -> str:
        """Add classified bug to vector store"""
        metadata = {
            'type': 'bug',
            'label': label,
            'reason': reason,
            'team': team or '',
            'severity': severity or '',
            'timestamp': datetime.now().isoformat(),
            'text': bug_text
        }
        
        vectorstore = self.get_vectorstore()
        ids = vectorstore.add_texts(
            texts=[bug_text],
            metadatas=[metadata],
            ids=[bug_id] if bug_id else None
        )
        
        return ids[0]
    
    def get_dynamic_few_shot_examples(
        self,
        bug_text: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Get dynamic few-shot examples from vector store
        
        Searches for similar bugs with type='example' in metadata
        """
        try:
            # Search with filter for examples only
            results = self.similarity_search(
                query=bug_text,
                k=top_k * 2,  # Get more to filter
                filter={"type": "example"}
            )
            
            # Format results
            examples = []
            for r in results:
                meta = r.get('metadata', {})
                if meta.get('type') == 'example' and meta.get('label'):
                    examples.append({
                        'id': r.get('id'),
                        'description': r.get('text', ''),
                        'label': meta.get('label')
                    })
            
            return examples[:top_k]
        except Exception as e:
            logger.error(f"❌ Failed to get dynamic examples: {e}")
            return []
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        try:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            if PINECONE_HOST:
                index = pc.Index(PINECONE_INDEX_NAME, host=PINECONE_HOST)
            else:
                index = pc.Index(PINECONE_INDEX_NAME)
            
            # Get index stats
            stats = index.describe_index_stats()
            
            return {
                "available": True,
                "index": PINECONE_INDEX_NAME,
                "total_vectors": stats.total_vector_count if hasattr(stats, 'total_vector_count') else 0,
                "dimension": 768
            }
        except Exception as e:
            logger.error(f"❌ Failed to get vector store stats: {e}")
            return {
                "available": False,
                "error": str(e)
            }
    
    # ========================
    # LLM Providers
    # ========================
    
    def get_gpt_llm(self, temperature: float = 0.0) -> ChatOpenAI:
        """Get GPT-5 LLM via LangChain ChatOpenAI"""
        if self.gpt_llm is None:
            logger.info(f"Creating ChatOpenAI: {OPENAI_MODEL_NAME}")
            self.gpt_llm = ChatOpenAI(
                model=OPENAI_MODEL_NAME,
                openai_api_key=OPENAI_API_KEY,
                openai_api_base=OPENAI_API_BASE,
                temperature=temperature,
                model_kwargs={"response_format": {"type": "json_object"}} if not OPENAI_MODEL_NAME.startswith("gpt-5") else {}
            )
        return self.gpt_llm
    
    def get_llama_provider(self) -> LlamaProvider:
        """Get Llama provider"""
        if self.llama_provider is None:
            logger.info("Creating Llama provider")
            self.llama_provider = LlamaProvider()
            self.llama_provider.load_model()  # Pre-load for performance
        return self.llama_provider
    
    # ========================
    # Classification Chains
    # ========================
    
    def _build_classification_prompt(
        self,
        labels: List[str],
        label_descriptions: str,
        example_text: str = "",
        is_batch: bool = False
    ) -> ChatPromptTemplate:
        """Build classification prompt template"""
        
        if is_batch:
            system_msg = """Bạn là chuyên gia QA với 10+ năm kinh nghiệm, chuyên phân tích và phân loại bug hàng loạt.

=== NHIỆM VỤ ===
Phân loại TẤT CẢ các bug trong danh sách. Mỗi bug phải được gán ĐÚNG MỘT nhãn.

=== NGỮ CẢNH ===
Các nhãn phân loại:
{label_descriptions}

=== LẬP LUẬN ===
Với mỗi bug:
1. Đọc toàn bộ thông tin (có thể chứa nhiều trường)
2. TỰ ĐỘNG xác định trường chứa nội dung bug chính
3. BỎ QUA thông tin không liên quan (ID, ngày tạo, v.v.)
4. Xác định từ khóa chính từ mô tả lỗi
5. So khớp với nhãn phù hợp nhất
6. Đánh giá severity: Critical (crash/security) > High (major function) > Medium (poor UX) > Low (minor display)

=== QUY TẮC ===
- PHẢI phân loại HẾT tất cả bugs
- KHÔNG bỏ sót bug nào
- KHÔNG tạo nhãn mới ngoài danh sách
- Lý do ngắn gọn (< 30 từ) bằng tiếng Việt

Danh sách bugs (format [index]: text):
{bug_list}

Trả về JSON array với {count} objects, mỗi object có: index, label, reason, team, severity."""
            
            human_msg = "Phân loại tất cả {count} bugs trên."
            
        else:
            system_msg = """Bạn là chuyên gia QA với 10+ năm kinh nghiệm, chuyên phân tích và phân loại bug.

=== NHIỆM VỤ ===
Phân loại bug report vào CHÍNH XÁC MỘT nhãn phù hợp nhất.

=== NGỮ CẢNH ===
Các nhãn phân loại:
{label_descriptions}

Ví dụ bugs đã phân loại:
{example_text}

=== LẬP LUẬN ===
1. Đọc toàn bộ thông tin bug (có thể chứa nhiều trường: No, Summary, Description, Priority, Status, v.v.)
2. TỰ ĐỘNG XÁC ĐỊNH trường nào chứa nội dung mô tả bug chính
3. BỎ QUA thông tin không liên quan (ID, ngày, người báo cáo, v.v.)
4. Tập trung vào mô tả lỗi để xác định từ khóa chính
5. So sánh với các nhãn, tìm nhãn khớp nhất về ngữ nghĩa
6. Nếu nhiều nhãn phù hợp, ưu tiên nhãn cụ thể hơn (VD: "Backend" > "Functional")
7. Đánh giá severity: Critical (system crash/security) > High (major function broken) > Medium (poor UX) > Low (minor display issue)

=== QUY TẮC ===
- KHÔNG tạo nhãn mới ngoài danh sách
- Lý do phải ngắn gọn (< 30 từ) và bằng tiếng Việt
- TỰ ĐỘNG lọc thông tin quan trọng

Bug report (có thể chứa nhiều trường, hãy tự động lọc):
<<<
{bug_description}
>>>

Trả về JSON object với: label, reason, team, severity."""
            
            human_msg = "Phân loại bug trên."
        
        return ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("human", human_msg)
        ])
    
    async def classify_with_gpt(
        self,
        description: str,
        labels: List[str],
        label_descriptions: str,
        example_text: str = "",
        team_groups: List[str] = None,
        retries: int = 3
    ) -> Dict:
        """Classify single bug using GPT-5 with LangChain"""
        logger.info("\n" + "="*80)
        logger.info("🤖 GPT-5 CLASSIFY (LangChain)")
        logger.info(f"📝 Input: {description[:100]}...")
        
        # Build chain
        prompt = self._build_classification_prompt(
            labels=labels,
            label_descriptions=label_descriptions,
            example_text=example_text,
            is_batch=False
        )
        
        llm = self.get_gpt_llm()
        
        # Use structured output with retry
        llm_with_structure = llm.with_structured_output(BugClassification)
        chain = prompt | llm_with_structure
        
        # Retry logic
        last_exc = None
        for attempt in range(1, retries + 1):
            try:
                result = await chain.ainvoke({
                    "label_descriptions": label_descriptions,
                    "example_text": example_text,
                    "bug_description": description
                })
                
                # Convert Pydantic model to dict
                result_dict = {
                    "label": result.label,
                    "reason": result.reason,
                    "team": result.team,
                    "severity": result.severity
                }
                
                logger.info(f"✅ {result_dict['label']} - {result_dict.get('team', 'N/A')}")
                logger.info("="*80 + "\n")
                return result_dict
                
            except Exception as e:
                last_exc = e
                wait = 0.5 * (2 ** (attempt - 1))
                logger.warning(f"⚠️ Attempt {attempt} failed, retrying in {wait}s...")
                await asyncio.sleep(wait)
        
        logger.error(f"❌ All attempts failed: {last_exc}")
        logger.info("="*80 + "\n")
        return {
            "label": labels[0] if labels else "",
            "reason": f"Error: {str(last_exc)}",
            "team": None,
            "severity": None
        }
    
    async def classify_with_llama(
        self,
        description: str,
        labels: List[str],
        examples: List[Dict] = None
    ) -> Dict:
        """Classify single bug using Llama"""
        logger.info("\n" + "="*80)
        logger.info("🦙 LLAMA CLASSIFY")
        logger.info(f"📝 Input: {description[:100]}...")
        
        llama = self.get_llama_provider()
        
        labels_text = ", ".join(labels)
        
        # Optimized prompt from original llama_service (working version)
        prompt = f"""<|start_header_id|>system<|end_header_id|>

You are an expert QA bug classifier with 10+ years of experience. Analyze bug reports and classify them into the correct category. Always respond with valid JSON only.<|eot_id|><|start_header_id|>user<|end_header_id|>

=== TASK ===
Classify the following bug report into ONE category from the available list.

=== ANALYSIS STEPS ===
1. Read all information in the bug report (may contain multiple fields: No, Summary, Description, Priority, Status, etc.).
2. AUTOMATICALLY IDENTIFY which field contains the main bug description (usually Summary, Description, or similar fields).
3. IGNORE irrelevant information (such as ID, Create date, Reporter, etc.).
4. Focus on the error description to identify main keywords.
5. Match with available labels and choose the most appropriate one semantically.
6. Assess severity based on actual impact: Critical (system crash/security) > High (main function broken) > Medium (poor experience) > Low (minor display issues).

=== RULES ===
- DO NOT make up new labels outside the list.
- Reason must be concise (< 30 words) and in Vietnamese.
- AUTOMATICALLY filter important information from input data.

Available categories: {labels_text}

Bug report (may contain multiple fields, automatically filter important information):
<<<
{description}
>>>

Respond with ONLY a JSON object (no additional text):
{{
  "label": "the most appropriate category from the list",
  "reason": "concise explanation in Vietnamese (15-25 words)",
  "severity": "Low, Medium, High, or Critical"
}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{{"""
        
        try:
            response = await llama.agenerate(prompt, max_tokens=200, temperature=0.3)
            
            # Parse JSON
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()
            
            if not response.startswith('{'):
                response = '{' + response
            
            # Extract JSON
            json_start = response.find('{')
            if json_start >= 0:
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
                    result = json.loads(json_str)
                    
                    # Validate label
                    label = result.get('label', '')
                    if label not in labels:
                        label_lower = label.lower()
                        for valid_label in labels:
                            if valid_label.lower() in label_lower or label_lower in valid_label.lower():
                                label = valid_label
                                break
                    
                    final_result = {
                        'label': label if label in labels else labels[0],
                        'reason': result.get('reason', 'Classified by Llama'),
                        'team': result.get('team'),
                        'severity': result.get('severity')
                    }
                    
                    logger.info(f"✅ {final_result['label']}")
                    logger.info("="*80 + "\n")
                    return final_result
            
            # Fallback: keyword matching
            for label in labels:
                if label.lower() in response.lower():
                    logger.warning("⚠️ Using keyword fallback")
                    logger.info("="*80 + "\n")
                    return {
                        'label': label,
                        'reason': 'Classified by keyword matching (Llama)',
                        'team': None,
                        'severity': None
                    }
            
            logger.warning("⚠️ Using default label")
            logger.info("="*80 + "\n")
            return {
                'label': labels[0],
                'reason': 'Default classification (Llama)',
                'team': None,
                'severity': None
            }
        
        except Exception as e:
            logger.error(f"❌ Llama error: {e}")
            logger.info("="*80 + "\n")
            return {
                'label': labels[0] if labels else '',
                'reason': f'Error: {str(e)}',
                'team': None,
                'severity': None
            }
    
    async def batch_classify_with_gpt(
        self,
        descriptions: List[str],
        indexes: List[int],
        labels: List[str],
        label_descriptions: str,
        example_text: str = "",
        retries: int = 4
    ) -> Dict[int, Dict]:
        """Batch classify bugs using GPT-5 with LangChain"""
        logger.info("\n" + "="*80)
        logger.info(f"🤖 GPT-5 BATCH CLASSIFY (LangChain) - Count: {len(descriptions)}")
        
        # Build bug list
        bug_list = "\n".join([f"[{indexes[i]}]: {descriptions[i]}" for i in range(len(descriptions))])
        
        # Build chain
        prompt = self._build_classification_prompt(
            labels=labels,
            label_descriptions=label_descriptions,
            example_text=example_text,
            is_batch=True
        )
        
        llm = self.get_gpt_llm()
        llm_with_structure = llm.with_structured_output(BatchClassificationResult)
        chain = prompt | llm_with_structure
        
        # Retry logic
        last_exc = None
        for attempt in range(1, retries + 1):
            try:
                result = await chain.ainvoke({
                    "label_descriptions": label_descriptions,
                    "bug_list": bug_list,
                    "count": len(descriptions)
                })
                
                # Convert to dict mapping
                results_dict = {}
                for item in result.classifications:
                    results_dict[item.index] = {
                        "label": item.label,
                        "reason": item.reason,
                        "team": item.team,
                        "severity": item.severity
                    }
                
                logger.info(f"✅ Batch complete: {len(results_dict)} results")
                logger.info("="*80 + "\n")
                return results_dict
                
            except Exception as e:
                last_exc = e
                wait = 0.6 * (2 ** (attempt - 1))
                logger.warning(f"⚠️ Attempt {attempt} failed, retrying in {wait}s...")
                await asyncio.sleep(wait)
        
        logger.error(f"❌ Batch failed: {last_exc}")
        logger.info("="*80 + "\n")
        return {}
    
    async def batch_classify_with_llama(
        self,
        descriptions: List[str],
        labels: List[str],
        examples: List[Dict] = None
    ) -> List[Dict]:
        """Batch classify bugs using Llama"""
        logger.info("\n" + "="*80)
        logger.info(f"🦙 LLAMA BATCH CLASSIFY - Count: {len(descriptions)}")
        
        llama = self.get_llama_provider()
        labels_text = ", ".join(labels)
        bugs_list = "\n".join([f"{i+1}. {desc}" for i, desc in enumerate(descriptions)])
        
        # Optimized batch prompt from original llama_service (working version)
        prompt = f"""<|start_header_id|>system<|end_header_id|>

You are an expert QA bug classifier with 10+ years of experience. Analyze multiple bug reports and classify each into the correct category. Always respond with valid JSON array only.<|eot_id|><|start_header_id|>user<|end_header_id|>

=== TASK ===
Classify ALL {len(descriptions)} bug reports below into categories.
Each bug must be assigned EXACTLY ONE label, with reason, team, and severity.

=== ANALYSIS STEPS ===
For each bug:
1. Read all information (may contain multiple fields like No, Summary, Description, Priority, etc.).
2. AUTOMATICALLY IDENTIFY which field contains the main bug description.
3. IGNORE irrelevant information (ID, create date, reporter, etc.).
4. Identify main keywords from the error description.
5. Match with available labels and choose the most appropriate one.
6. Prioritize specific labels (e.g., "Database" > "Backend" if related to queries).
7. Assess severity based on actual impact.

=== RULES ===
- MUST classify ALL bugs (include all indexes in the list).
- DO NOT skip any bug.
- DO NOT make up new labels outside the list.
- Reason must be concise (< 30 words) and in Vietnamese.
- AUTOMATICALLY filter important information from input data.

Available categories: {labels_text}

Bug reports to classify (each bug may contain multiple fields, automatically filter important information):
{bugs_list}

Respond with ONLY a JSON array containing {len(descriptions)} objects (no additional text). Each object must have:
- "label": category from the list
- "reason": brief explanation in Vietnamese (15-25 words)
- "severity": Low, Medium, High, or Critical

JSON array:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

["""
        
        try:
            max_tokens = min(300 + (len(descriptions) * 100), 2000)
            response = await llama.agenerate(prompt, max_tokens=max_tokens, temperature=0.3)
            
            # Parse JSON array
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()
            
            if not response.startswith('['):
                response = '[' + response
            
            array_start = response.find('[')
            if array_start >= 0:
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
                    results_array = json.loads(json_str)
                    
                    if isinstance(results_array, list):
                        processed = []
                        for i, result in enumerate(results_array):
                            if i >= len(descriptions):
                                break
                            
                            label = result.get('label', '')
                            if label not in labels:
                                label_lower = label.lower()
                                for valid_label in labels:
                                    if valid_label.lower() in label_lower or label_lower in valid_label.lower():
                                        label = valid_label
                                        break
                            
                            processed.append({
                                'label': label if label in labels else labels[0],
                                'reason': result.get('reason', 'Classified by Llama batch'),
                                'team': result.get('team'),
                                'severity': result.get('severity')
                            })
                        
                        # Fill missing
                        while len(processed) < len(descriptions):
                            processed.append({
                                'label': labels[0],
                                'reason': 'Batch incomplete',
                                'team': None,
                                'severity': None
                            })
                        
                        logger.info(f"✅ Batch complete: {len(processed)} results")
                        logger.info("="*80 + "\n")
                        return processed[:len(descriptions)]
            
            # Fallback: individual classification
            logger.warning("⚠️ Batch parsing failed, falling back to individual")
            logger.info("="*80 + "\n")
            results = []
            for desc in descriptions:
                result = await self.classify_with_llama(desc, labels, examples)
                results.append(result)
            return results
            
        except Exception as e:
            logger.error(f"❌ Batch error: {e}")
            logger.info("="*80 + "\n")
            # Fallback
            results = []
            for desc in descriptions:
                try:
                    result = await self.classify_with_llama(desc, labels, examples)
                    results.append(result)
                except:
                    results.append({
                        'label': labels[0],
                        'reason': 'Error during fallback',
                        'team': None,
                        'severity': None
                    })
            return results
    
    # ========================
    # Unified Interface
    # ========================
    
    async def classify_bug(
        self,
        description: str,
        labels: List[str],
        model: Literal["GPT-5", "Llama"] = "GPT-5",
        label_descriptions: str = "",
        example_text: str = "",
        team_groups: List[str] = None,
        examples: List[Dict] = None
    ) -> Dict:
        """
        Unified interface for bug classification.
        Routes to appropriate model (GPT-5 or Llama).
        """
        if model == "GPT-5":
            return await self.classify_with_gpt(
                description=description,
                labels=labels,
                label_descriptions=label_descriptions,
                example_text=example_text,
                team_groups=team_groups
            )
        elif model == "Llama":
            return await self.classify_with_llama(
                description=description,
                labels=labels,
                examples=examples
            )
        else:
            raise ValueError(f"Unsupported model: {model}")
    
    async def batch_classify(
        self,
        descriptions: List[str],
        labels: List[str],
        model: Literal["GPT-5", "Llama"] = "GPT-5",
        indexes: List[int] = None,
        label_descriptions: str = "",
        example_text: str = "",
        examples: List[Dict] = None
    ) -> Any:
        """
        Unified interface for batch classification.
        Returns: Dict[int, Dict] for GPT-5, List[Dict] for Llama
        """
        if model == "GPT-5":
            if indexes is None:
                indexes = list(range(len(descriptions)))
            return await self.batch_classify_with_gpt(
                descriptions=descriptions,
                indexes=indexes,
                labels=labels,
                label_descriptions=label_descriptions,
                example_text=example_text
            )
        elif model == "Llama":
            return await self.batch_classify_with_llama(
                descriptions=descriptions,
                labels=labels,
                examples=examples
            )
        else:
            raise ValueError(f"Unsupported model: {model}")


# ========================
# Global Singleton
# ========================

_unified_service = None

def get_unified_service() -> UnifiedLLMService:
    """Get singleton unified service instance"""
    global _unified_service
    if _unified_service is None:
        _unified_service = UnifiedLLMService()
    return _unified_service


# ========================
# Convenience Functions (Backward Compatibility)
# ========================

def get_embeddings() -> OpenAIEmbeddings:
    """Get embeddings instance"""
    return get_unified_service().get_embeddings()

def get_vectorstore() -> PineconeVectorStore:
    """Get vector store instance"""
    return get_unified_service().get_vectorstore()

def similarity_search(query: str, k: int = 5, filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Perform similarity search"""
    return get_unified_service().similarity_search(query, k, filter)

def add_bug_to_vectorstore(
    bug_text: str,
    label: str,
    reason: str,
    team: Optional[str] = None,
    severity: Optional[str] = None,
    bug_id: Optional[str] = None
) -> str:
    """Add classified bug to vector store"""
    return get_unified_service().add_bug_to_vectorstore(
        bug_text=bug_text,
        label=label,
        reason=reason,
        team=team,
        severity=severity,
        bug_id=bug_id
    )

def get_dynamic_few_shot_examples(
    bug_text: str,
    top_k: int = 5,
    use_local_embeddings: bool = False
) -> List[Dict[str, Any]]:
    """Get dynamic few-shot examples from vector store (backward compatibility)"""
    return get_unified_service().get_dynamic_few_shot_examples(bug_text, top_k)

def get_vector_store_stats(use_local_embeddings: bool = False) -> Dict[str, Any]:
    """Get vector store statistics (backward compatibility)"""
    return get_unified_service().get_vector_store_stats()
