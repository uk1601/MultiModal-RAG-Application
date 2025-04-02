
import os
import uuid
import pickle
import json
import boto3
from typing import List, Dict
from fastapi import HTTPException
from llama_index.core import SummaryIndex, StorageContext, load_index_from_storage
from llama_index.core.schema import TextNode
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from pinecone import Pinecone, ServerlessSpec
from pinecone import Index
from llama_parse import LlamaParse

from app.models.report_models import ReportOutput
from app.services import rag_service


# create stored_pages.json file


class ReportService:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.LLAMA_PARSED_DIR = "./llama_parsed"
        os.makedirs(self.LLAMA_PARSED_DIR, exist_ok=True)
        self.embedding_dimension = 1536
        self.index_name = "multimodalindex"
        self.STORED_PAGES_FILE = "stored_pages.json"
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.embed_model = OpenAIEmbedding(api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small")
        self.llm = OpenAI(model="gpt-4o", system_prompt="You are a report generation assistant...")
        self.pinecone_index = self.setup_pinecone_index()
        self.stored_pages = self.load_stored_pages()
        self.output_dir = Path("output_reports")
        self.output_dir.mkdir(exist_ok=True)


    def setup_pinecone_index(self) -> Index:
        indexes = self.pc.list_indexes()
        if any(index['name'] == self.index_name for index in indexes.get('indexes', [])):
            pinecone_index = self.pc.Index(self.index_name)
        else:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            pinecone_index = self.pc.Index(self.index_name)
        return pinecone_index

    def download_pdf_from_s3(self, pdf_name: str) -> str:
        temp_dir = "/tmp/"
        pdf_path = os.path.join(temp_dir, f"{os.path.basename(pdf_name)}")
        try:
            self.s3_client.download_file("cfapublications", pdf_name, pdf_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error downloading PDF from S3: {str(e)}")
        return pdf_path

    def parse_document(self, pdf_path: str) -> List[dict]:
        parsed_file = os.path.join(self.LLAMA_PARSED_DIR, os.path.basename(pdf_path) + ".json")
        if os.path.exists(parsed_file):
            with open(parsed_file, "rb") as f:
                parsed_data = pickle.load(f)
        else:
            parser = LlamaParse(
                result_type="markdown",
                use_vendor_multimodal_model=True,
                vendor_multimodal_model_name="openai-gpt-4o-mini",
                api_key=os.getenv("LLAMA_PARSE_API_KEY"),
                fast_mode=True,
                continuous_mode=True,
                guess_xlsx_sheet_names=True,
                annotate_links=True
            )
            parsed_data = parser.get_json_result(pdf_path)
            os.makedirs("data_images", exist_ok=True)
            os.makedirs(self.LLAMA_PARSED_DIR, exist_ok=True)
            parser.get_images(parsed_data, download_path="data_images")
            with open(parsed_file, "wb") as f:
                pickle.dump(parsed_data, f)
        return parsed_data

    def store_in_pinecone(self, parsed_data: List[dict]):
        try:
            for page_data in parsed_data:
                pdf_name = page_data.get("file_path")
                pdf_name = pdf_name.split("/")[-1]
                for page in page_data.get("pages", []):
                    page_num = page.get("page")
                    page_text = page.get("text", "")
                    # sample page: {'page': 1, 'text': 'CFA INSTITUTE RESEARCH FOUNDATION / MONOGRAPH\n\nBEYOND ACTIVE AND\nPASSIVE INVESTING\nTHE CUSTOMIZATION OF FINANCE\n\nMARC R. REINGANUM\nKENNETH A. BLAY', 'images': [{'name': 'img_p0_1.png', 'height': 975, 'width': 1305, 'x': -1.1901414, 'y': 226.47601319999995, 'original_width': 1224, 'original_height': 914, 'path': 'data_images/acfe6bc1-c81c-4203-add4-4c853c356985-img_p0_1.png', 'job_id': 'acfe6bc1-c81c-4203-add4-4c853c356985', 'original_file_path': 'downloaded_pdfs/Beyond Active and Passive Investing_ The Customization of Finance.pdf', 'page_number': 1}], 'items': [], 'status': 'OK', 'links': []}
                    # extract only name of images from it an store in metdata
                    images = page.get("images", [])
                    if len(images) > 0:
                        page_images = [img.get("path") for img in images]
                        page_text += "Use these images for markdown" + "\n".join(page_images)
                    metadata = {
                        "page_num": page_num,
                        "pdf_name": pdf_name,
                        "text": page_text,
                    }
                    if (pdf_name, page_num) in self.stored_pages:
                        continue
                    if not page_text:
                        return
                    embedding = self.embed_model._get_text_embedding(page_text)
                    self.pinecone_index.upsert([{
                        "id": str(uuid.uuid4()),
                        "values": embedding,
                        "metadata": metadata
                    }])
                    self.stored_pages.add((pdf_name, page_num))
                    self.save_stored_pages()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error storing data in Pinecone: {str(e)}")

    def load_stored_pages(self) -> set:
        if os.path.exists(self.STORED_PAGES_FILE):
            with open(self.STORED_PAGES_FILE, "r") as f:
                return set(tuple(page) for page in json.load(f))
        return set()

    def save_stored_pages(self):
        with open(self.STORED_PAGES_FILE, "w") as f:
            json.dump([list(page) for page in self.stored_pages], f)

    def generate_structured_report(self, parsed_data: List[dict]) -> str:
        storage_dir = "storage_nodes_summary"
        text_nodes = self.get_text_nodes(parsed_data[0]["pages"], image_dir="data_images")
        if not os.path.exists(storage_dir):
            index = SummaryIndex(text_nodes)
            index.set_index_id("summary_index")
            index.storage_context.persist(storage_dir)
        else:
            storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
            index = load_index_from_storage(storage_context, index_id="summary_index")
        system_prompt = (
            "You are a report generation assistant. Generate a brief, single-paragraph summary "
            "that highlights key financial insights and any relevant visuals, keeping it under 100 words."
        )
        llm = OpenAI(model="gpt-4o", system_prompt=system_prompt)
        sllm = llm.as_structured_llm(output_cls=ReportOutput)
        query_engine = index.as_query_engine(
            similarity_top_k=5,
            llm=sllm,
            response_mode="compact"
        )

        response = query_engine.query("Generate a concise financial summary.")
        markdown_output = self.process_response_to_markdown(response)
        return markdown_output

    def get_text_nodes(self, parsed_pages, image_dir=None):
        text_nodes = []
        image_files = self._get_sorted_image_files(image_dir) if image_dir else []
        combined_text = "\n".join(page.get("md", "") for page in parsed_pages)
        metadata = {"page_num": 1, "parsed_text_markdown": combined_text}
        if image_files:
            metadata["image_path"] = str(image_files[0])
        node = TextNode(text=combined_text, metadata=metadata)
        text_nodes.append(node)
        return text_nodes

    def _get_sorted_image_files(self, image_dir):
        raw_files = [f for f in list(Path(image_dir).iterdir()) if f.is_file()]
        return sorted(raw_files, key=self.get_page_number)

    def get_page_number(self, file_name):
        import re
        match = re.search(r"-page-(\d+)\.jpg$", str(file_name))
        return int(match.group(1)) if match else 0

    def process_response_to_markdown(self, response_instance) -> str:

        markdown_output = f"## Report Summary\n\n---\n\n"
        markdown_output += "### Source Details\n\n"
        for idx, node in enumerate(response_instance.source_nodes, start=1):
            metadata = node.node.metadata
            markdown_output += f"#### Page {metadata.get('page_num', idx)}\n\n"
            main_text = metadata.get("parsed_text_markdown", "").strip()
            if main_text:
                markdown_output += f"{main_text}\n\n"
            image_path = metadata.get("image_path")
            if image_path:
                markdown_output += f"![Image for Page {metadata['page_num']}]({image_path})\n\n"
            markdown_output += "---\n\n"
        return markdown_output

    def convert_markdown_to_pdf(self, markdown_content: str) -> str:
        import markdown
        import pdfkit
        config = pdfkit.configuration(wkhtmltopdf='/opt/homebrew/Caskroom/wkhtmltopdf/0.12.6-2/')
        html_content = markdown.markdown(markdown_content)
        output_path = self.output_dir / "report_summary.pdf"
        pdfkit.from_string(html_content, str(output_path), configuration=config)
        return str(output_path)

    def generate_report(self, pdf_name: str) -> str:
        pdf_path = self.download_pdf_from_s3(pdf_name)
        parsed_data = self.parse_document(pdf_path)
        self.store_in_pinecone(parsed_data)
        markdown_output = self.generate_structured_report(parsed_data)
        report_pdf_path = self.convert_markdown_to_pdf(markdown_output)
        return report_pdf_path


import os
import json
import logging
import pickle
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm
from urllib.parse import urlparse
import requests
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential


class ProcessingCache:
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "processing_status.json"
        self.status = self.load_cache()

    def load_cache(self) -> Dict:
        if self.cache_file.exists():
            with open(self.cache_file, "r") as f:
                return json.load(f)
        return {
            "downloaded": [],
            "parsed": [],
            "embedded": [],
            "report_generated": [],
            "failed": {}
        }

    def save_cache(self):
        with open(self.cache_file, "w") as f:
            json.dump(self.status, f, indent=2)

    def mark_downloaded(self, url: str):
        if url not in self.status["downloaded"]:
            self.status["downloaded"].append(url)
            self.save_cache()

    def mark_parsed(self, url: str):
        if url not in self.status["parsed"]:
            self.status["parsed"].append(url)
            self.save_cache()

    def mark_embedded(self, url: str):
        if url not in self.status["embedded"]:
            self.status["embedded"].append(url)
            self.save_cache()

    def mark_report_generated(self, url: str):
        if url not in self.status["report_generated"]:
            self.status["report_generated"].append(url)
            self.save_cache()

    def mark_failed(self, url: str, stage: str, error: str):
        self.status["failed"][url] = {"stage": stage, "error": str(error)}
        self.save_cache()

    def is_completed(self, url: str) -> bool:
        return url in self.status["report_generated"]

    def is_downloaded(self, url: str) -> bool:
        return url in self.status["downloaded"]

    def is_parsed(self, url: str) -> bool:
        return url in self.status["parsed"]

    def is_embedded(self, url: str) -> bool:
        return url in self.status["embedded"]

    def is_report_generated(self, url: str) -> bool:
        return url in self.status["report_generated"]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def download_file(url: str, download_dir: Path) -> Path:
    response = requests.get(url, stream=True)
    response.raise_for_status()

    filename = os.path.basename(urlparse(url).path)
    file_path = download_dir / filename

    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return file_path


def setup_directories():
    dirs = {
        'download': Path("downloaded_pdfs"),
        'reports_md': Path("reports/markdown"),
        'reports_pdf': Path("reports/pdf"),
        'logs': Path("logs")
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


def get_report_paths(filename: str, dirs: Dict[str, Path]) -> Dict[str, Path]:
    base_name = Path(filename).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        'markdown': dirs['reports_md'] / f"{base_name}_{timestamp}.md",
        'pdf': dirs['reports_pdf'] / f"{base_name}_{timestamp}.pdf"
    }


def main(s3_urls: List[str], skip_parsing = True):
    # Setup logging
    dirs = setup_directories()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(dirs['logs'] / 'pdf_processing.log'),
            logging.StreamHandler()
        ]
    )

    # Initialize cache and ReportService
    cache = ProcessingCache()
    report_service = ReportService()
    logging.info(f"{report_service.STORED_PAGES_FILE}")

    # Process each URL
    for url in tqdm(s3_urls, desc="Processing PDFs"):
        if cache.is_completed(url):
            logging.info(f"Skipping {url} - already processed")
            continue

        filename = os.path.basename(urlparse(url).path)
        report_paths = get_report_paths(filename, dirs)

        try:
            # Download phase
            if not cache.is_downloaded(url):
                logging.info(f"Downloading {url}")
                pdf_path = download_file(url, dirs['download'])
                cache.mark_downloaded(url)
            else:
                pdf_path = dirs['download'] / filename

            # Parsing phase
            if not cache.is_parsed(url):
                logging.info(f"Parsing {url}")
                parsed_data = report_service.parse_document(str(pdf_path))
                cache.mark_parsed(url)
            else:
                parsed_file = Path(report_service.LLAMA_PARSED_DIR) / f"{pdf_path.name}.json"
                with open(parsed_file, "rb") as f:
                    parsed_data = pickle.load(f)
            # logging.info(f"Successfully parsed \n {parsed_data[0]}")
            # Embedding and storage phase
            # if not cache.is_embedded(url):
            if True:
                logging.info(f"-------------------Creating embeddings and storing in Pinecone for {url}---------------------")
                report_service.store_in_pinecone(parsed_data)
                cache.mark_embedded(url)

            # Report generation phase
            if not cache.is_report_generated(url):
                logging.info(f"Generating report for {url}")

                # Generate markdown report
                markdown_output = report_service.generate_structured_report(parsed_data)
                with open(report_paths['markdown'], 'w', encoding='utf-8') as f:
                    f.write(markdown_output)

                # Convert to PDF
                pdf_path = report_service.convert_markdown_to_pdf(markdown_output)
                # Move the PDF to the reports directory
                os.rename(pdf_path, report_paths['pdf'])

                cache.mark_report_generated(url)

            logging.info(f"Successfully processed {url}")

        except Exception as e:
            error_msg = f"Error processing {url}: {str(e)}"
            logging.error(error_msg)
            # print stack trace as well
            import traceback
            traceback.print_exc()

            cache.mark_failed(url, "processing", error_msg)
            continue

    # Final summary
    logging.info("\nProcessing Summary:")
    logging.info(f"Total URLs: {len(s3_urls)}")
    logging.info(f"Successfully processed: {len(cache.status['report_generated'])}")
    logging.info(f"Failed: {len(cache.status['failed'])}")
    if cache.status['failed']:
        logging.info("\nFailed URLs and reasons:")
        for url, details in cache.status['failed'].items():
            logging.info(f"{url}: {details['error']}")





def generate_summary_report(urls: List[str], cache: ProcessingCache, dirs: Dict[str, Path]):
    """Generate an overall summary report of the processing run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_md_path = dirs['reports_md'] / f"processing_summary_{timestamp}.md"

    summary_content = [
        "# PDF Processing Summary Report",
        f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\nTotal PDFs processed: {len(urls)}",
        f"\nSuccessfully completed: {len(cache.status['report_generated'])}",
        f"\nFailed: {len(cache.status['failed'])}",
        "\n## Processing Details",
    ]

    for url in urls:
        filename = os.path.basename(urlparse(url).path)
        status = "✅ Completed" if cache.is_completed(url) else "❌ Failed"
        failure_details = cache.status['failed'].get(url, {})

        summary_content.extend([
            f"\n### {filename}",
            f"- Status: {status}",
            f"- URL: {url}",
            "- Processing stages completed:",
            f"  - Downloaded: {'✓' if cache.is_downloaded(url) else '✗'}",
            f"  - Parsed: {'✓' if cache.is_parsed(url) else '✗'}",
            f"  - Embedded: {'✓' if cache.is_embedded(url) else '✗'}",
            f"  - Report Generated: {'✓' if cache.is_report_generated(url) else '✗'}"
        ])

        if failure_details:
            summary_content.extend([
                "- Failure details:",
                f"  - Stage: {failure_details.get('stage', 'unknown')}",
                f"  - Error: {failure_details.get('error', 'unknown')}"
            ])

    # Write summary report
    with open(summary_md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_content))

    # Convert to PDF
    report_service = ReportService()
    summary_pdf_path = report_service.convert_markdown_to_pdf('\n'.join(summary_content))
    os.rename(summary_pdf_path, dirs['reports_pdf'] / f"processing_summary_{timestamp}.pdf")



