import pathlib,sys,unittest
import jinja2

class TemplatePolicy(unittest.TestCase):
    def test_explicit_off_wins_over_historical_markers(self):
        source=(pathlib.Path(__file__).resolve().parents[1]/'scripts/windows-harness/qwen-sharp-v22.5.0.jinja').read_text(encoding='utf-8')
        template=jinja2.Environment().from_string(source)
        for marker in ['<|think_xhigh|>','<|think_on|>','<|think_medium|>']:
            for enable,effort in [(False,'medium'),(True,'none')]:
                result=template.render(messages=[{'role':'user','content':'historical '+marker}],tools=[],add_generation_prompt=True,enable_thinking=enable,reasoning_effort=effort)
                self.assertTrue(result.endswith('<think>\n\n</think>\n\n'))
        result=template.render(messages=[{'role':'user','content':'hello'}],tools=[],add_generation_prompt=True,enable_thinking=True)
        self.assertTrue(result.endswith('<think>\n'))

if __name__=='__main__':unittest.main()
