import unittest

from tts_processor import preprocess_all, strip_markdown


class TestMarkdownStripping(unittest.TestCase):
    def test_bold_markers_removed(self):
        self.assertEqual(strip_markdown("**Bold** text"), "Bold text")

    def test_italic_markers_removed(self):
        self.assertEqual(strip_markdown("*italic* text"), "italic text")

    def test_heading_markers_removed(self):
        self.assertEqual(strip_markdown("## Sub heading"), "Sub heading")

    def test_list_markers_removed(self):
        self.assertEqual(strip_markdown("- item one\n- item two"), "item one\nitem two")

    def test_ordered_list_markers_removed(self):
        self.assertEqual(strip_markdown("1. first\n2. second"), "first\nsecond")

    def test_task_checkbox_removed(self):
        self.assertEqual(strip_markdown("- [ ] todo\n- [x] done"), "todo\ndone")

    def test_blockquote_marker_removed(self):
        self.assertEqual(strip_markdown("> quoted line"), "quoted line")

    def test_strikethrough_removed(self):
        self.assertEqual(strip_markdown("~~strike~~ text"), "strike text")

    def test_code_fence_removed_but_content_kept(self):
        self.assertEqual(strip_markdown("```python\nprint(1)\n```"), "print(1)")

    def test_horizontal_rule_removed(self):
        self.assertEqual(strip_markdown("---"), "")
        self.assertEqual(strip_markdown("***"), "")

    def test_link_keeps_label_and_url(self):
        out = strip_markdown("See [Dashboard](https://grafana.example.com/d/abc)")
        self.assertEqual(out, "See Dashboard, https://grafana.example.com/d/abc")

    def test_link_whose_label_is_the_url_is_not_duplicated(self):
        out = strip_markdown("[https://example.com](https://example.com)")
        self.assertEqual(out, "https://example.com")

    def test_image_removed(self):
        self.assertEqual(strip_markdown("![alt](img.png)"), "")

    def test_table_rendered_as_labelled_sentences(self):
        text = "| Host | Status | CPU |\n|------|--------|-----|\n| srv1 | up | 12% |"
        self.assertEqual(strip_markdown(text), "Host: srv1, Status: up, CPU: 12%.")

    def test_table_without_outer_pipes(self):
        text = "Host | Status\n-----|------\nsrv1 | up"
        self.assertEqual(strip_markdown(text), "Host: srv1, Status: up.")

    def test_table_separator_row_removed(self):
        self.assertEqual(strip_markdown("|---|---|"), "")

    def test_no_markdown_characters_reach_piper(self):
        text = "**Bold** ## Heading\n- item\n| Host | Status |\n|---|---|\n| srv1 | up |"
        out = strip_markdown(text)
        for char in ("*", "#", "|", "`", "~"):
            self.assertNotIn(char, out)

    def test_markdown_is_stripped_by_preprocess_all(self):
        out = preprocess_all("**Bold** text")
        self.assertNotIn("*", out)
        self.assertIn("Bold", out)


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

    def test_triple_dash_silenced(self):
        text = "|---|---|---|"
        out = preprocess_all(text)
        self.assertNotIn("double dash", out)
        self.assertNotIn("dash", out)

    def test_four_dashes_silenced(self):
        text = "----"
        out = preprocess_all(text)
        self.assertNotIn("double dash", out)
        self.assertNotIn("dash", out)

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
