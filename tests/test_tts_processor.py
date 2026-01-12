import unittest

from tts_processor import preprocess_all


class TestTtsProcessor(unittest.TestCase):
    def test_url_normalization(self):
        text = "Check https://example.com/path"
        out = preprocess_all(text)
        self.assertIn("colon", out)
        self.assertIn("forward slash", out)
        self.assertIn("dot", out)

    def test_email_normalization(self):
        text = "Alert admin@example.com"
        out = preprocess_all(text)
        self.assertIn(" at ", out)
        self.assertIn(" dot ", out)

    def test_domain_optional_plural(self):
        text = "Check domain(s) now"
        out = preprocess_all(text)
        self.assertIn("domain or domains", out)

    def test_number_hyphenation(self):
        text = "showing 4 out of 24"
        out = preprocess_all(text)
        self.assertIn("twenty four", out)

    def test_mail_port_regression(self):
        text = "mail.mahadeva.co.uk - SMTP endpoint on port 25, Agent: Berlin - Germany"
        out = preprocess_all(text)
        self.assertIn("port twenty five", out)

    def test_double_dash(self):
        text = "Use --testmode for debug"
        out = preprocess_all(text)
        self.assertIn("double dash", out)

    def test_slash_or(self):
        text = "Use this/that option"
        out = preprocess_all(text)
        self.assertIn("this or that", out)

    def test_sha(self):
        text = "Hash is SHA256"
        out = preprocess_all(text)
        self.assertIn("sha two five six", out)


if __name__ == "__main__":
    unittest.main()
