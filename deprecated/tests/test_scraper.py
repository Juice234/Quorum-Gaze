import base64
import tempfile
import unittest
from pathlib import Path

from quorum_scraper.scraper import save_view_image


class FakeResponse:
    status = 200


class FakeCanvas:
    def __init__(self, image_bytes):
        self.first = self
        self.image_bytes = image_bytes

    async def wait_for(self, **_options):
        return None

    async def evaluate(self, _script):
        encoded = base64.b64encode(self.image_bytes).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"


class FakePage:
    def __init__(self, image_bytes):
        self.canvas = FakeCanvas(image_bytes)
        self.opened_url = None

    async def goto(self, url, **_options):
        self.opened_url = url
        return FakeResponse()

    def locator(self, selector):
        self.asserted_selector = selector
        return self.canvas

    async def wait_for_function(self, _script, **_options):
        return None


class ScraperTests(unittest.IsolatedAsyncioTestCase):
    async def test_view_image_is_saved_from_the_canvas(self):
        image_bytes = b"jpeg from the BOTB canvas"
        page = FakePage(image_bytes)

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "clean.jpg"
            await save_view_image(page, "https://example.com/viewimage", output, 1000)

            self.assertEqual(output.read_bytes(), image_bytes)
            self.assertEqual(page.asserted_selector, "canvas")
            self.assertEqual(page.opened_url, "https://example.com/viewimage")


if __name__ == "__main__":
    unittest.main()
