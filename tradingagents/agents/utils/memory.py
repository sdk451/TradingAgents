import chromadb
from chromadb.config import Settings
from openai import OpenAI
import os
import asyncio
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sentence_transformers import SentenceTransformer

# Maximum payload size for Google embeddings (36KB limit, use 30KB to be safe)
MAX_EMBEDDING_PAYLOAD_SIZE = 30000


class FinancialSituationMemory:
    def __init__(self, name, config):
        self.config = config
        self.provider = config.get("llm_provider", "openai").lower()
        if self.provider == "openai":
            self.embedding = "text-embedding-3-small"
            self.client = OpenAI(base_url=config["backend_url"])
            self.embedding_model = None
        elif self.provider == "google":
            import asyncio
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.set_event_loop(asyncio.new_event_loop())
            google_api_key = os.getenv("GOOGLE_API_KEY")
            self.embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
            self.client = None
        elif self.provider == "anthropic":
            self.embedding_model = None
            self.client = None
        else:
            # Use a local embedding model for other non-OpenAI providers
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.client = None
        self.chroma_client = chromadb.Client(Settings(allow_reset=True))
        self.situation_collection = self.chroma_client.create_collection(name=name)

    def _truncate_for_embedding(self, text: str, max_size: int = MAX_EMBEDDING_PAYLOAD_SIZE) -> str:
        """
        Intelligently truncate text for embedding while preserving semantic meaning.
        For concatenated reports, preserves structure by keeping portions of each section.
        
        Args:
            text: Text to truncate
            max_size: Maximum size in bytes
            
        Returns:
            Truncated text that preserves key information
        """
        # Convert to bytes to check size
        text_bytes = text.encode('utf-8')
        
        if len(text_bytes) <= max_size:
            return text
        
        # Strategy: For concatenated reports (separated by \n\n), preserve structure
        # by keeping the beginning and end of each section
        sections = text.split('\n\n')
        
        if len(sections) > 1:
            # Multiple sections (likely concatenated reports)
            # Keep first part of each section and last part of last section
            truncated_sections = []
            section_budget = max_size // len(sections)
            
            for i, section in enumerate(sections):
                section_bytes = section.encode('utf-8')
                if len(section_bytes) <= section_budget:
                    truncated_sections.append(section)
                else:
                    # Keep beginning (summary) and end (conclusion) of section
                    # Use 60% for beginning, 40% for end
                    begin_size = int(section_budget * 0.6)
                    end_size = section_budget - begin_size
                    
                    begin_bytes = section_bytes[:begin_size]
                    end_bytes = section_bytes[-end_size:] if end_size > 0 else b''
                    
                    try:
                        begin = begin_bytes.decode('utf-8')
                        end = end_bytes.decode('utf-8') if end_bytes else ''
                    except UnicodeDecodeError:
                        begin = begin_bytes.decode('utf-8', errors='ignore')
                        end = end_bytes.decode('utf-8', errors='ignore') if end_bytes else ''
                    
                    if end:
                        truncated_sections.append(f"{begin}\n[... truncated ...]\n{end}")
                    else:
                        truncated_sections.append(f"{begin}\n[... truncated ...]")
            
            result = '\n\n'.join(truncated_sections)
            # Final check - if still too large, fall back to simple truncation
            if len(result.encode('utf-8')) > max_size:
                return self._simple_truncate(text, max_size)
            return result
        else:
            # Single section - use simple truncation with beginning and end
            return self._simple_truncate(text, max_size)
    
    def _simple_truncate(self, text: str, max_size: int) -> str:
        """
        Simple truncation keeping beginning and end of text.
        
        Args:
            text: Text to truncate
            max_size: Maximum size in bytes
            
        Returns:
            Truncated text
        """
        text_bytes = text.encode('utf-8')
        if len(text_bytes) <= max_size:
            return text
        
        # Keep 60% for beginning, 40% for end
        begin_size = int(max_size * 0.6)
        end_size = max_size - begin_size
        
        begin_bytes = text_bytes[:begin_size]
        end_bytes = text_bytes[-end_size:] if end_size > 0 else b''
        
        try:
            begin = begin_bytes.decode('utf-8')
            end = end_bytes.decode('utf-8') if end_bytes else ''
        except UnicodeDecodeError:
            begin = begin_bytes.decode('utf-8', errors='ignore')
            end = end_bytes.decode('utf-8', errors='ignore') if end_bytes else ''
        
        if end:
            return f"{begin}\n\n[... content truncated ...]\n\n{end}"
        return f"{begin}\n\n[... content truncated ...]"

    def get_embedding(self, text):
        # Truncate text for Google provider to avoid payload size errors
        if self.provider == "google":
            text = self._truncate_for_embedding(text)
            return self.embedding_model.embed_query(text)
        elif self.provider == "openai":
            response = self.client.embeddings.create(
                model=self.embedding, input=text
            )
            return response.data[0].embedding
        elif self.provider == "anthropic":
            raise NotImplementedError("Memory features are currently not supported for Anthropic provider. Please use OpenAI or Google for memory-enabled workflows.")
        else:
            # Use local embedding model
            return self.embedding_model.encode(text).tolist()

    def add_situations(self, situations_and_advice):
        """Add financial situations and their corresponding advice. Parameter is a list of tuples (situation, rec)"""

        situations = []
        advice = []
        ids = []
        embeddings = []

        offset = self.situation_collection.count()

        for i, (situation, recommendation) in enumerate(situations_and_advice):
            situations.append(situation)
            advice.append(recommendation)
            ids.append(str(offset + i))
            embeddings.append(self.get_embedding(situation))

        self.situation_collection.add(
            documents=situations,
            metadatas=[{"recommendation": rec} for rec in advice],
            embeddings=embeddings,
            ids=ids,
        )

    def get_memories(self, current_situation, n_matches=1):
        """Find matching recommendations using provider-appropriate embeddings"""
        query_embedding = self.get_embedding(current_situation)

        results = self.situation_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_matches,
            include=["metadatas", "documents", "distances"],
        )

        matched_results = []
        for i in range(len(results["documents"][0])):
            matched_results.append(
                {
                    "matched_situation": results["documents"][0][i],
                    "recommendation": results["metadatas"][0][i]["recommendation"],
                    "similarity_score": 1 - results["distances"][0][i],
                }
            )

        return matched_results


if __name__ == "__main__":
    # Example usage
    matcher = FinancialSituationMemory()

    # Example data
    example_data = [
        (
            "High inflation rate with rising interest rates and declining consumer spending",
            "Consider defensive sectors like consumer staples and utilities. Review fixed-income portfolio duration.",
        ),
        (
            "Tech sector showing high volatility with increasing institutional selling pressure",
            "Reduce exposure to high-growth tech stocks. Look for value opportunities in established tech companies with strong cash flows.",
        ),
        (
            "Strong dollar affecting emerging markets with increasing forex volatility",
            "Hedge currency exposure in international positions. Consider reducing allocation to emerging market debt.",
        ),
        (
            "Market showing signs of sector rotation with rising yields",
            "Rebalance portfolio to maintain target allocations. Consider increasing exposure to sectors benefiting from higher rates.",
        ),
    ]

    # Add the example situations and recommendations
    matcher.add_situations(example_data)

    # Example query
    current_situation = """
    Market showing increased volatility in tech sector, with institutional investors 
    reducing positions and rising interest rates affecting growth stock valuations
    """

    try:
        recommendations = matcher.get_memories(current_situation, n_matches=2)

        for i, rec in enumerate(recommendations, 1):
            print(f"\nMatch {i}:")
            print(f"Similarity Score: {rec['similarity_score']:.2f}")
            print(f"Matched Situation: {rec['matched_situation']}")
            print(f"Recommendation: {rec['recommendation']}")

    except Exception as e:
        print(f"Error during recommendation: {str(e)}")
