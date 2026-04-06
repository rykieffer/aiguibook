
import sys
import os
import json
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestAiguibookPipeline(unittest.TestCase):
    
    def test_01_imports(self):
        """Check basic imports."""
        try:
            from audiobook_ai.analysis.character_analyzer import SpeechTag, EMOTION_INSTRUCTIONS_FR
            print("Imports OK")
        except Exception as e:
            self.fail("Import failed: %s" % e)

    def test_02_load_analysis_json(self):
        """Simulate loading analysis.json and ensure Tags are objects."""
        from audiobook_ai.analysis.character_analyzer import SpeechTag, EMOTION_INSTRUCTIONS_FR
        
        analysis_path = "/home/hermes/aiguibook_analysis.json"
        if not os.path.exists(analysis_path):
            self.skipTest("Analysis JSON not found")
            
        state = {"tags": {}, "chars": [], "analyzed": False}
        
        # Simulate GUI Load Logic (The Fix)
        with open(analysis_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        tags_dict = data.get("tags", {})
        chars = data.get("chars", [])
        
        # FIX: Convert to SpeechTags
        tags_objects = {}
        for sid, t in tags_dict.items():
             speaker = t.get("speaker", "narrator")
             char_name = t.get("char")
             emo = t.get("emotion", "neutral")
             
             if emo not in EMOTION_INSTRUCTIONS_FR:
                 emo = "neutral"
             
             instr = EMOTION_INSTRUCTIONS_FR[emo]
             
             tags_objects[sid] = SpeechTag(
                 segment_id=sid, speaker_type=speaker, 
                 character_name=char_name, emotion=emo, 
                 voice_id="narrator", emotion_instruction=instr
             )
        
        state["tags"] = tags_objects
        state["chars"] = chars
        state["analyzed"] = True
        
        self.assertTrue(len(tags_objects) > 0)
        first_tag = list(tags_objects.values())[0]
        self.assertTrue(hasattr(first_tag, 'emotion'))
        print("JSON Load OK: %d tags" % len(tags_objects))

    def test_03_generation_logic(self):
        """Simulate Generation Loop data access."""
        from audiobook_ai.analysis.character_analyzer import SpeechTag, EMOTION_INSTRUCTIONS_FR
        
        # Create dummy tag (Object)
        tag_obj = SpeechTag(
            segment_id="obj_1", speaker_type="dialogue", character_name="Joan",
            emotion="angry", voice_id="narrator", emotion_instruction="Parlez avec colère"
        )
        # Create dummy tag (Dict - legacy JSON)
        tag_dict = {
            "speaker": "narrator", "char": "Barry", "emotion": "calm"
        }
        
        tags = {"tag1": tag_obj, "tag2": tag_dict}
        
        # Simulate Generation Loop Logic (The Fix)
        for sid, t in tags.items():
            if hasattr(t, 'emotion'):
                # Object mode
                emotion = t.emotion
                speaker = t.speaker_type
            else:
                # Dict mode
                emotion = t.get("emotion", "neutral")
                speaker = t.get("speaker", "narrator")
            
            emo_instr = EMOTION_INSTRUCTIONS_FR.get(emotion, EMOTION_INSTRUCTIONS_FR["neutral"])
            self.assertIsNotNone(emo_instr)
            
        print("Generation Loop Logic OK")

    def test_04_ui_calculation(self):
        """Verify table data counts."""
        from audiobook_ai.analysis.character_analyzer import SpeechTag
        
        tags = {
            "s1": SpeechTag(segment_id="s1", speaker_type="d", character_name="Joan", emotion="angry", voice_id="n", emotion_instruction=""),
            "s2": SpeechTag(segment_id="s2", speaker_type="d", character_name="Joan", emotion="calm", voice_id="n", emotion_instruction=""),
        }
        chars = ["Joan"]
        
        table = []
        for c in chars:
            c_tags = [t for t in tags.values() if hasattr(t, 'character_name') and t.character_name == c]
            count = len(c_tags)
            emos = list(set([t.emotion for t in c_tags]))
            table.append([c, count, ", ".join(emos)])
            
        self.assertEqual(table[0][1], 2)
        print("UI Table Calculation OK")

if __name__ == "__main__":
    unittest.main(verbosity=2)
