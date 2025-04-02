import hashlib
import os
import json
import logging
import io
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union, Sequence
from datetime import datetime
from dataclasses import dataclass, asdict
import llama_index
import markdown
import pandas as pd
import pdfkit
import torch
import numpy as np
import cv2
from PIL import Image
import pymupdf as fitz
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from pydantic import BaseModel, Field
from llama_index.core.response_synthesizers.type import ResponseMode
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.auto import partition

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table
from reportlab.lib import colors
import os
from pinecone import Pinecone, ServerlessSpec
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    Settings,
    load_index_from_storage
)
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.schema import NodeWithScore
from llama_parse import LlamaParse

llama_index.core.set_global_handler(
    "arize_phoenix", endpoint="https://llamatrace.com/v1/traces"
)
embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-base-en-v1.5",
            embed_batch_size=10
)
Settings.embed_model = embed_model
Settings.dimension = 768
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('document_processing.log'),
        logging.StreamHandler()
    ]
)


class TextBlock(BaseModel):
    """Text block in the report"""
    content_type: str = Field("text", description="Type of content block")  # Set default value
    text: str = Field(..., description="The text content")
    source: Optional[str] = Field(None, description="Source of the text")
    page_num: Optional[int] = Field(None, description="Page number")
    metadata: Optional[Dict] = Field(None, description="Additional metadata")


class ImageBlock():
    """Image block in the report"""
    content_type: str = Field("image", description="Type of content block")
    file_path: str = Field(..., description="Path to the image file")
    caption: Optional[str] = Field(None, description="Image caption")
    source: Optional[str] = Field(None, description="Source of the image")
    page_num: Optional[int] = Field(None, description="Page number")
    metadata: Optional[Dict] = Field(None, description="Additional metadata")


class TableBlock():
    """Table block in the report"""
    content_type: str = Field("table", description="Type of content block")
    file_path: str = Field(..., description="Path to the table image")
    excel_path: Optional[str] = Field(None, description="Path to Excel file")
    caption: Optional[str] = Field(None, description="Table caption")
    source: Optional[str] = Field(None, description="Source of the table")
    page_num: Optional[int] = Field(None, description="Page number")
    metadata: Optional[Dict] = Field(None, description="Additional metadata")


class ReportOutput():
    """Structured output for report generation"""
    title: str = Field(..., description="Report title")
    summary: str = Field(..., description="Executive summary")
    blocks: List[Union[TextBlock, ImageBlock, TableBlock]] = Field(
        ...,
        description="Content blocks in the report"
    )


def initialize_vector_store(index_name: str = "document-store") -> PineconeVectorStore:
    """Initialize Pinecone vector store with the new client."""

    # Initialize Pinecone client
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

    # Check if index exists, if not create it
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=768,  # OpenAI embedding dimension
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

    # Initialize vector store
    vector_store = PineconeVectorStore(
        index_name=index_name,
        environment=os.environ.get("PINECONE_ENVIRONMENT")
    )

    return vector_store


import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from pydantic import BaseModel, Field
import fitz
import torch
from transformers import AutoTokenizer, AutoModel
from IPython.display import display, Markdown, Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ContentBlock(BaseModel):
    """Base content block"""
    content_type: str
    page_num: int
    metadata: Dict


class TextBlock(ContentBlock):
    """Text block for output"""
    text: str = Field(..., description="The text content")


class ImageBlock(ContentBlock):
    """Image block for output"""
    image_path: str = Field(..., description="Path to image file")
    caption: Optional[str] = Field(None, description="Image caption")


class ReportOutput(BaseModel):
    """Structured report output"""
    title: str = Field(..., description="Report title")
    summary: str = Field(..., description="Executive summary")
    blocks: List[Union[TextBlock, ImageBlock]] = Field(
        ..., description="Content blocks"
    )

    def render(self):
        """Render report with text and images"""
        display(Markdown(f"# {self.title}"))
        display(Markdown(f"## Executive Summary\n{self.summary}"))

        for block in self.blocks:
            if isinstance(block, TextBlock):
                display(Markdown(block.text))
            else:
                display(Image(filename=block.image_path))
                if block.caption:
                    display(Markdown(f"*{block.caption}*"))


class DocumentProcessor:
    """Document processing with PyMuPDF"""

    def __init__(self, storage_dir: str = "./storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

        # Create subdirectories
        self.image_dir = self.storage_dir / "images"
        self.image_dir.mkdir(exist_ok=True)

        self.table_dir = self.storage_dir / "tables"
        self.table_dir.mkdir(exist_ok=True)

        self.cache_dir = self.storage_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)

        self.logger = logging.getLogger(__name__)

    def process_document(self, file_path: str) -> List[Document]:
        """Process document and return list of Documents"""
        cache_path = self.cache_dir / f"{Path(file_path).stem}_processed.pkl"

        if cache_path.exists():
            self.logger.info(f"Loading cached processing results: {cache_path}")
            import pickle
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        try:
            doc = fitz.open(file_path)
            all_documents = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                page_documents = self._process_page(page, page_num, file_path)
                all_documents.extend(page_documents)

            doc.close()

            # Cache results
            with open(cache_path, 'wb') as f:
                import pickle
                pickle.dump(all_documents, f)

            return all_documents

        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            raise

    def _process_page(self, page, page_num: int, source_file: str) -> List[Document]:
        """Process a single page and return list of Documents"""
        documents = []

        try:
            # Extract text blocks
            text_blocks = self._extract_text_blocks(page, page_num)
            for idx, block in enumerate(text_blocks):
                bboxstr = f"{block['bbox']['x0']},{block['bbox']['y0']},{block['bbox']['x1']},{block['bbox']['y1']}"
                doc = Document(
                    text=block["text"],
                    metadata={
                        "source": f"{Path(source_file).stem}-page{page_num}-text{idx}",
                        "page_num": page_num,
                        "type": "text",
                        "bbox": bboxstr,
                        "id_": f"{Path(source_file).stem}-page{page_num}-text{idx}"
                    }
                )
                documents.append(doc)

            # Extract images
            images = self._extract_images(page, page_num, source_file)
            for idx, image in enumerate(images):
                bbox = image.metadata["bbox"]
                # bboxstr = f"{bbox['x1']},{bbox['y1']},{bbox['x2']},{bbox['y2']}"
                doc = Document(
                    text=f"Image from page {page_num}",
                    metadata={
                        "source": f"{Path(source_file).stem}-page{page_num}-image{idx}",
                        "page_num": page_num,
                        "type": "image",
                        "image_path": image.metadata["image_path"],
                        "bbox": bbox,
                        "id_": f"{Path(source_file).stem}-page{page_num}-image{idx}"
                    }
                )
                documents.append(doc)

            # Extract tables
            tables = self._extract_tables(page, page_num, source_file)
            for idx, table in enumerate(tables):


                doc = Document(
                    text=f"Table from page {page_num}",
                    metadata={
                        "source": f"{Path(source_file).stem}-page{page_num}-table{idx}",
                        "page_num": page_num,
                        "type": "table",
                        "table_path": table.metadata["table_path"],
                        "excel_path": table.metadata["excel_path"],
                        "bbox": table.metadata["bbox"],
                        "id_": f"{Path(source_file).stem}-page{page_num}-table{idx}"
                    }
                )
                documents.append(doc)

            return documents

        except Exception as e:
            self.logger.error(f"Error processing page {page_num}: {str(e)}")
            raise

    def _extract_text_blocks(self, page, page_num: int) -> List[Dict]:
        """Extract text blocks from page"""
        blocks = []
        for block in page.get_text("blocks"):
            if block[6] == 0:  # Text block
                blocks.append({
                    "text": block[4],
                    "bbox": {
                        "x0": block[0],
                        "y0": block[1],
                        "x1": block[2],
                        "y1": block[3]
                    },
                    "page_num": page_num
                })
        return blocks

    def _extract_images(self, page, page_num: int, source_file: str) -> List[Dict]:
        """Extract images from page with descriptive text"""
        images = []
        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base_image = page.parent.extract_image(xref)

                if base_image:
                    image_path = self.image_dir / f"{Path(source_file).stem}_page{page_num}_img{img_index}.png"
                    with open(image_path, "wb") as f:
                        f.write(base_image["image"])

                    # Get image location and surrounding text
                    image_info = page.get_image_info(xrefs=[xref])[0]
                    bbox = image_info['bbox']

                    # Get text before and after image for context
                    before_text, after_text = self._extract_text_around_item(
                        page.get_text("blocks"),
                        bbox,
                        page.rect.height
                    )

                    # Generate image description
                    image_description = self._generate_image_description(
                        image_path,
                        before_text,
                        after_text
                    )

                    images.append(Document(
                        text=image_description,  # Use descriptive text for indexing
                        metadata={
                            "source": str(Path(source_file).stem),
                            "type": "image",
                            "image_path": str(image_path),
                            "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                            "page_num": page_num,
                            "id": f"img_{img_index}",
                            "description": image_description  # Store description in metadata
                        }
                    ))

            except Exception as e:
                self.logger.warning(f"Error extracting image: {str(e)}")
                continue

        return images

    def _extract_tables(self, page, page_num: int, source_file: str) -> []:
        """Extract tables from page with descriptive text"""
        tables = []
        try:
            found_tables = page.find_tables(
                horizontal_strategy="lines_strict",
                vertical_strategy="lines_strict"
            )

            for table_idx, table in enumerate(found_tables):
                try:
                    if not table.header.external:
                        # Convert to pandas DataFrame
                        pandas_df = table.to_pandas()

                        # Save table as Excel
                        excel_path = self.table_dir / f"{Path(source_file).stem}_page{page_num}_table{table_idx}.xlsx"
                        pandas_df.to_excel(str(excel_path))

                        # Create image of table area
                        table_bbox = fitz.Rect(table.bbox)
                        table_pix = page.get_pixmap(clip=table_bbox)
                        image_path = self.table_dir / f"{Path(source_file).stem}_page{page_num}_table{table_idx}.png"
                        table_pix.save(str(image_path))

                        # Get surrounding text
                        before_text, after_text = self._extract_text_around_item(
                            page.get_text("blocks"),
                            table_bbox,
                            page.rect.height
                        )

                        # Generate table description
                        table_description = self._generate_table_description(
                            pandas_df,
                            before_text,
                            after_text,
                            table.header.names
                        )

                        tables.append(Document(
                            text=table_description,  # Use descriptive text for indexing
                            metadata={
                                "source": str(Path(source_file).stem),
                                "type": "table",
                                "table_path": str(image_path),
                                "excel_path": str(excel_path),
                                "bbox": f"{table_bbox[0]},{table_bbox[1]},{table_bbox[2]},{table_bbox[3]}",
                                "page_num": page_num,
                                "id": f"table_{table_idx}",
                                "description": table_description,  # Store description in metadata
                                "columns": list(pandas_df.columns)
                            }
                        ))

                except Exception as e:
                    self.logger.warning(f"Error processing table {table_idx} on page {page_num}: {str(e)}")
                    continue

        except Exception as e:
            self.logger.warning(f"Error extracting tables from page {page_num}: {str(e)}")

        return tables

    def _extract_text_around_item(self, blocks: List[Dict], bbox: fitz.Rect, page_height: float) -> Tuple[str, str]:
        """Extract text before and after an item (image/table)"""
        before_text = ""
        after_text = ""

        for block in blocks:
            if block[6] == 0:  # Text block
                block_bbox = fitz.Rect(block[:4])

                # Text above the item
                if block_bbox.y1 < bbox.y0 and abs(block_bbox.y1 - bbox.y0) < 100:
                    before_text += block[4] + " "

                # Text below the item
                if block_bbox.y0 > bbox.y1 and abs(block_bbox.y0 - bbox.y1) < 100:
                    after_text += block[4] + " "

        return before_text.strip(), after_text.strip()

    def _generate_image_description(self, image_path: str, before_text: str, after_text: str) -> str:
        """Generate descriptive text for an image"""
        try:
            # Use image analysis to generate description
            image = Image.open(image_path)

            # Combine contextual text with image content
            description = "This image shows "

            if before_text:
                description += f"content related to: {before_text}. "

            # Add basic image properties
            width, height = image.size
            description += f"The image dimensions are {width}x{height} pixels. "

            if after_text:
                description += f"Additional context: {after_text}"

            return description

        except Exception as e:
            self.logger.warning(f"Error generating image description: {str(e)}")
            return "Image content"

    def _generate_table_description(
            self,
            df: pd.DataFrame,
            before_text: str,
            after_text: str,
            header_names: List[str]
    ) -> str:
        """Generate descriptive text for a table"""
        try:
            description = "This table contains "

            # Add context from surrounding text
            if before_text:
                description += f"information about {before_text}. "

            # Add table structure information
            num_rows, num_cols = df.shape
            description += f"It has {num_rows} rows and {num_cols} columns. "
            description += f"The columns are: {', '.join(header_names)}. "

            # Add summary statistics if numerical columns exist
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                description += "Summary statistics: "
                for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                    mean = df[col].mean()
                    description += f"{col} (average: {mean:.2f}), "

            if after_text:
                description += f"Additional context: {after_text}"

            return description

        except Exception as e:
            self.logger.warning(f"Error generating table description: {str(e)}")
            return "Table content"
    #
    # def _extract_images(self, page, page_num: int, source_file: str) -> List[Dict]:
    #     """Extract images from page"""
    #     images = []
    #     for img_index, img in enumerate(page.get_images(full=True)):
    #         try:
    #             xref = img[0]
    #             base_image = page.parent.extract_image(xref)
    #
    #             if base_image:
    #                 image_path = self.image_dir / f"{Path(source_file).stem}_page{page_num}_img{img_index}.png"
    #                 with open(image_path, "wb") as f:
    #                     f.write(base_image["image"])
    #                 image_info = page.get_image_info(xrefs=[xref])[0]
    #                 bbox = image_info['bbox']
    #                 logger.info(f"Extracted image from page {page_num}: {image_path}")
    #                 images.append(Document(
    #                     text=f"Image from page {page_num}",
    #                     metadata={
    #                         "source": str(Path(source_file).stem),
    #                         "type": "image",
    #                         "image_path": str(image_path),
    #                         "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
    #                         "page_num": page_num,
    #                         "id": f"img_{img_index}"
    #                     }
    #                 ))
    #         except Exception as e:
    #             self.logger.warning(f"Error extracting image: {str(e)}")
    #             continue
    #
    #     return images
    #
    # def _extract_tables(self, page, page_num: int, source_file: str) -> List[Dict]:
    #     """Extract tables from page"""
    #     tables = []
    #     try:
    #         found_tables = page.find_tables(
    #             horizontal_strategy="lines_strict",
    #             vertical_strategy="lines_strict"
    #         )
    #
    #         for table_idx, table in enumerate(found_tables):
    #             try:
    #                 if not table.header.external:
    #                     # Convert to pandas DataFrame
    #                     pandas_df = table.to_pandas()
    #
    #                     # Save table as Excel
    #                     excel_path = self.table_dir / f"{Path(source_file).stem}_page{page_num}_table{table_idx}.xlsx"
    #                     pandas_df.to_excel(str(excel_path))
    #
    #                     # Create image of table area
    #                     table_bbox = fitz.Rect(table.bbox)
    #                     table_pix = page.get_pixmap(clip=table_bbox)
    #                     image_path = self.table_dir / f"{Path(source_file).stem}_page{page_num}_table{table_idx}.png"
    #                     table_pix.save(str(image_path))
    #                     bboxstr = f"{table.bbox[0]},{table.bbox[1]},{table.bbox[2]},{table.bbox[3]}"
    #                     tables.append(Document(metadata={
    #                         "table_path": str(image_path),
    #                         "excel_path": str(excel_path),
    #                         "bbox": bboxstr,
    #                         "columns": list(pandas_df.columns),
    #                         "id_": table_idx,
    #                     }))
    #
    #             except Exception as e:
    #                 self.logger.warning(f"Error processing table {table_idx} on page {page_num}: {str(e)}")
    #                 continue
    #
    #     except Exception as e:
    #         self.logger.warning(f"Error extracting tables from page {page_num}: {str(e)}")
    #
    #     return tables

class RAGQueryEngine:
    """Enhanced RAG Query Engine with Pinecone vector store"""

    def __init__(
            self,
            storage_dir: str = "./storage",
            pinecone_api_key: str = os.getenv("PINECONE_API_KEY"),
            pinecone_environment: str = "gcp-starter",
            index_name: str = "rag-index"
    ):
        self.storage_dir = Path(storage_dir)
        self.cache_dir = self.storage_dir / "cache"
        self.logger = logging.getLogger(__name__)

        # Create directories
        self.storage_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)

        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = index_name

        # Initialize embedding model
        self.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-base-en-v1.5",
            embed_batch_size=10
        )

        # Initialize LLM
        system_prompt = """You are a report generation assistant that produces well-formatted
        responses with text, images, and tables. Include relevant visual elements when they add value.
        Structure your response according to the ReportOutput schema."""

        self.llm = OpenAI(
            model="gpt-4o-mini",
            api_base="https://models.inference.ai.azure.com",
            system_prompt=system_prompt
        )

        # Set global models
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm

        # Initialize or load Pinecone index
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        """Initialize or load Pinecone vector store"""
        try:
            # Check if index exists
            existing_indexes = self.pc.list_indexes().names()

            if self.index_name not in existing_indexes:
                # Create new index
                self.pc.create_index(
                    name=self.index_name,
                    dimension=768,  # BGE-base dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                self.logger.info(f"Created new Pinecone index: {self.index_name}")

            # Connect to index
            self.pinecone_index = self.pc.Index(self.index_name)

            # Initialize vector store
            self.vector_store = PineconeVectorStore(
                pinecone_index=self.pinecone_index,
                metadata_filters={"type": ["text", "image", "table"]}
            )


            # Initialize storage context
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )

        except Exception as e:
            self.logger.error(f"Error initializing vector store: {str(e)}")
            raise

    def _get_cache_key(self, documents: List[Document]) -> str:
        """Generate cache key for documents"""
        content_hash = hashlib.md5()
        for doc in sorted(documents, key=lambda x: x.id_):
            content_hash.update(str(doc.hash).encode())
        return content_hash.hexdigest()

    def build_index(self, documents: List[Document]):
        """Build or load index with caching"""
        try:
            # Generate cache key
            cache_key = self._get_cache_key(documents)
            cache_path = self.cache_dir / f"index_cache_{cache_key}.pkl"

            # Check cache
            if cache_path.exists():
                self.logger.info("Loading index from cache...")
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)

                # Verify if documents are already in Pinecone
                stats = self.pinecone_index.describe_index_stats()
                if stats.total_vector_count == len(documents):
                    self.logger.info("Documents already indexed in Pinecone")
                    return

            # Build new index
            self.logger.info("Building new index...")

            # Prepare documents for upsert
            vectors_to_upsert = []
            for doc in documents:
                # Generate embedding
                embedding = self.embed_model.get_text_embedding(doc.text)

                # Prepare vector data
                vector_data = {
                    "id": doc.id_,
                    "values": embedding,
                    "metadata": {
                        **doc.metadata,
                        "text": doc.text,
                        "doc_hash": doc.hash
                    }
                }
                vectors_to_upsert.append(vector_data)

            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                self.pinecone_index.upsert(vectors=batch)

            # Cache the index metadata
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'doc_hashes': [doc.hash for doc in documents],
                    'index_name': self.index_name
                }, f)

            self.logger.info(f"Successfully indexed {len(documents)} documents")

        except Exception as e:
            self.logger.error(f"Error building index: {str(e)}")
            raise

    async def query(
            self,
            query_text: str,
            response_mode: str = "tree_summarize",
            similarity_top_k: int = 10
    ) -> ReportOutput:
        """Execute structured query"""
        try:
            # Create structured LLM
            structured_llm = self.llm.as_structured_llm(output_cls=ReportOutput)

            # Configure response synthesizer
            response_synthesizer = get_response_synthesizer(
                response_mode=ResponseMode.TREE_SUMMARIZE,
                structured_answer_filtering=True
            )
            index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store, show_progress=True)
            # Create query engine
            query_engine = index.as_query_engine(
                similarity_top_k=similarity_top_k,
                response_synthesizer=response_synthesizer,
                llm=structured_llm,
                node_postprocessors=[
                    SimilarityPostprocessor(similarity_cutoff=0.3)
                ]
            )

            # Execute query
            response = await query_engine.aquery(query_text)

            # Process and structure the response
            return self._process_response(response)

        except Exception as e:
            self.logger.error(f"Error executing query: {str(e)}")
            raise

    def _process_response(self, response) -> ReportOutput:
        """Process query response into structured output with enhanced descriptions"""
        blocks = []

        for node in response.source_nodes:
            metadata = node.metadata

            if metadata.get("type") == "text":
                blocks.append(TextBlock(
                    text=node.text,
                    source=metadata.get("source"),
                    page_num=metadata.get("page_num"),
                    metadata=metadata,
                    content_type="text"
                ))

            elif metadata.get("type") == "image":
                blocks.append(ImageBlock(
                    file_path=metadata.get("image_path"),
                    caption=metadata.get("description"),  # Use generated description
                    source=metadata.get("source"),
                    page_num=metadata.get("page_num"),
                    metadata=metadata,
                    content_type="image"
                ))

            elif metadata.get("type") == "table":
                blocks.append(TableBlock(
                    file_path=metadata.get("table_path"),
                    excel_path=metadata.get("excel_path"),
                    caption=metadata.get("description"),  # Use generated description
                    source=metadata.get("source"),
                    page_num=metadata.get("page_num"),
                    metadata=metadata,
                    content_type="table"
                ))

        return ReportOutput(
            title="Report",
            summary=response.response,
            blocks=blocks
        )

    def render_report(self, report: ReportOutput, output_filename: str = "report.pdf"):
        """Generate report in two steps: markdown first, then PDF"""
        try:
            # Step 1: Generate and save markdown
            markdown_content = self._generate_markdown(report)
            markdown_path = output_filename.replace('.pdf', '.md')

            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            # Step 2: Convert markdown to PDF using ReportLab
            self._markdown_to_pdf(markdown_content, output_filename)

            self.logger.info(f"Report generated successfully: {output_filename}")

        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            raise

    def _generate_markdown(self, report: ReportOutput) -> str:
        """Generate markdown content with proper image and table handling"""
        markdown_parts = []

        # Add title and summary
        markdown_parts.extend([
            f"# {report.title}\n",
            "## Executive Summary\n",
            f"{report.summary}\n\n"
        ])

        # Process content blocks
        for block in report.blocks:
            try:
                if isinstance(block, TextBlock):
                    # Add text content with source information
                    source_info = f"### Content from {block.metadata["source"] if block.metadata["source"] else 'Unknown'}"
                    if block.page_num is not None:
                        source_info += f" (Page {block.page_num})"

                    markdown_parts.extend([
                        source_info,
                        f"{block.text}\n"
                    ])

                elif isinstance(block, ImageBlock):
                    # Add image content with proper path handling
                    if hasattr(block, 'file_path') and block.file_path and os.path.exists(block.file_path):
                        source_info = f"### Image from {block.source if block.source else 'Unknown'}"
                        if block.page_num is not None:
                            source_info += f" (Page {block.page_num})"

                        # Convert to absolute path for proper rendering
                        abs_path = os.path.abspath(block.file_path)

                        markdown_parts.extend([
                            source_info,
                            f"![Image]({abs_path})\n"
                        ])

                        if block.caption:
                            markdown_parts.append(f"*{block.caption}*\n")

                elif isinstance(block, TableBlock):
                    # Add table content with both image and Excel reference
                    if hasattr(block, 'file_path') and block.file_path and os.path.exists(block.file_path):
                        source_info = f"### Table from {block.source if block.source else 'Unknown'}"
                        if block.page_num is not None:
                            source_info += f" (Page {block.page_num})"

                        # Convert to absolute path for proper rendering
                        abs_path = os.path.abspath(block.file_path)

                        markdown_parts.extend([
                            source_info,
                            f"![Table]({abs_path})\n"
                        ])

                        if block.caption:
                            markdown_parts.append(f"*{block.caption}*\n")

                        if hasattr(block, 'excel_path') and block.excel_path and os.path.exists(block.excel_path):
                            excel_abs_path = os.path.abspath(block.excel_path)
                            markdown_parts.append(f"[Download Excel]({excel_abs_path})\n")

            except Exception as e:
                self.logger.error(f"Error processing block for markdown: {str(e)}")
                continue

        return "\n".join(markdown_parts)

    def _markdown_to_pdf(self, markdown_content: str, output_filename: str):
        """Convert markdown to PDF using ReportLab with proper image handling"""
        try:
            # Initialize document
            doc = SimpleDocTemplate(
                output_filename,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )

            # Define styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30
            )
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=16,
                spaceAfter=12
            )
            normal_style = ParagraphStyle(
                'CustomNormal',
                parent=styles['Normal'],
                fontSize=12,
                spaceAfter=12
            )
            caption_style = ParagraphStyle(
                'CustomCaption',
                parent=styles['Normal'],
                fontSize=10,
                textColor=colors.gray,
                spaceAfter=12,
                italics=True
            )

            # Build story (content)
            story = []

            # Process markdown content
            lines = markdown_content.split('\n')
            current_text = []

            for line in lines:
                if line.startswith('# '):  # Title
                    if current_text:
                        story.append(Paragraph('\n'.join(current_text), normal_style))
                        current_text = []
                    story.append(Paragraph(line[2:], title_style))

                elif line.startswith('## '):  # Section heading
                    if current_text:
                        story.append(Paragraph('\n'.join(current_text), normal_style))
                        current_text = []
                    story.append(Paragraph(line[3:], heading_style))

                elif line.startswith('### '):  # Subsection heading
                    if current_text:
                        story.append(Paragraph('\n'.join(current_text), normal_style))
                        current_text = []
                    story.append(Paragraph(line[4:], heading_style))

                elif line.startswith('!['):  # Image
                    if current_text:
                        story.append(Paragraph('\n'.join(current_text), normal_style))
                        current_text = []

                    # Extract image path
                    img_path = line[line.find('(') + 1:line.find(')')]

                    if os.path.exists(img_path):
                        # Get image dimensions
                        img = Image.open(img_path)
                        img_width, img_height = img.size
                        aspect = img_height / float(img_width)

                        # Calculate dimensions to fit page width
                        max_width = 6 * inch
                        width = min(max_width, img_width)
                        height = width * aspect

                        # Add image to story
                        img = RLImage(img_path, width=width, height=height)
                        story.append(img)
                        story.append(Spacer(1, 12))

                elif line.startswith('*'):  # Caption
                    if current_text:
                        story.append(Paragraph('\n'.join(current_text), normal_style))
                        current_text = []
                    story.append(Paragraph(line, caption_style))

                elif line.startswith('['):  # Link
                    if current_text:
                        story.append(Paragraph('\n'.join(current_text), normal_style))
                        current_text = []
                    story.append(Paragraph(line, normal_style))

                elif line.strip():  # Regular text
                    current_text.append(line)

                else:  # Empty line
                    if current_text:
                        story.append(Paragraph('\n'.join(current_text), normal_style))
                        current_text = []
                    story.append(Spacer(1, 12))

            # Add any remaining text
            if current_text:
                story.append(Paragraph('\n'.join(current_text), normal_style))

            # Build PDF
            doc.build(story)
            self.logger.info(f"PDF generated successfully: {output_filename}")

        except Exception as e:
            self.logger.error(f"Error converting to PDF: {str(e)}")
            raise




async def main():
    # Initialize query engine
    query_engine = RAGQueryEngine()
    processor = DocumentProcessor()
    # Load and process documents
    # documents = processor.process_document("/Users/saisuryamadhav/Documents/University/new/Assignment_3_Team1/backend/pdf/Beyond Active and Passive Investing_ The Customization of Finance.pdf")
    #
    # # Build index
    # query_engine.build_index(documents)

    # Execute query
    query = "Explain this with all the required images The Capital Asset Pricing Model Emerges?"
    # query = "What is this assignment about?"
    report = await query_engine.query(
        query,
        response_mode="tree_summarize",
        similarity_top_k=10
    )


    # Generate report
    query_engine.render_report(report, "financial_analysis.pdf")

import nest_asyncio
import asyncio

# Apply nest_asyncio to allow nested event loops (needed for Jupyter)
nest_asyncio.apply()
if __name__ == "__main__":
    asyncio.run(main())