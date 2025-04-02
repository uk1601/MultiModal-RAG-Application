from pydantic import BaseModel


class Publication(BaseModel):
    ID: int
    TITLE: str
    SUMMARY: str
    IMAGE_URL: str
    PDF_URL: str
    def __init__(self, id, title, summary, image_url, pdf_url):
        self.ID = id
        self.TITLE = title
        self.SUMMARY = summary
        self.IMAGE_URL = image_url
        self.PDF_URL = pdf_url

    def __repr__(self):
        return (f"Publication(ID={self.ID}, TITLE={self.TITLE}, SUMMARY={self.SUMMARY}, "
                f"IMAGE_URL={self.IMAGE_URL}, PDF_URL={self.PDF_URL})")
