from pydantic import BaseModel, Field
from typing import List

class TextBlock(BaseModel):
    text: str = Field(..., description="Text content.")

class ImageBlock(BaseModel):
    file_path: str = Field(..., description="Path to the image file.")

class ReportOutput(BaseModel):
    blocks: List[TextBlock | ImageBlock] = Field(..., description="List of text and image blocks.")

    def render(self) -> None:
        from IPython.display import display, Markdown, Image
        for block in self.blocks:
            if isinstance(block, TextBlock):
                display(Markdown(block.text))
            elif isinstance(block, ImageBlock):
                display(Image(filename=block.file_path))
