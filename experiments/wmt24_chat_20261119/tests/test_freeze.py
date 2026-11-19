"""No-network validation of WMT24 frozen source/ref correspondence and groups."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from freeze_test import TOTAL_EXPECTED,load_csv,freeze_rows,SOURCE_FIELDS,REFERENCE_FIELDS


def fake():
    source=[]
    gold=[]
    for i in range(TOTAL_EXPECTED):
        fr=i%2==0
        row={
            "source_language":"fr" if fr else "en",
            "target_language":"en" if fr else "fr",
            "source":("Bonjour monsieur numéro "+str(i) if fr else
                      "Hello, how can I help you today number "+str(i)),
            "doc_id":"conversation-"+str(i//30),
            "client_id":"client-"+str(i//30),
            "sender":"customer" if fr else "agent",
        }
        source.append(row)
        gold.append({**row,"reference":(
            "Hello sir number "+str(i) if fr else "Bonjour, comment puis-je vous aider "+str(i)
            ),"tags":"[]"})
    return source,gold


class TestWMT24Frozen(unittest.TestCase):
    def test_test_only_diverse_conversation_group_hashes(self):
        a,b=fake()
        result=freeze_rows(a,b)
        self.assertEqual(result["total_test_segments"],2091)
        self.assertGreaterEqual(result["conversation_groups"],60)
        self.assertGreaterEqual(result["french_to_english_segments"],600)
        self.assertFalse(result["model_evaluation_performed"])
        self.assertFalse(result["90_percent_human_adequacy_verified"])
        self.assertIn("NEVER train",result["test_policy"])
        self.assertEqual(len(result["group_ids_sha256"]),64)

    def test_human_reference_utterance_mismatch_rejected(self):
        a,b=fake()
        b[12]["source"]="totally different bilingual source"
        with self.assertRaisesRegex(ValueError,"Gold reference row mismatch"):
            freeze_rows(a,b)

    def test_missing_group_or_unknown_speaker_rejected(self):
        a,b=fake()
        a[4]["sender"]=b[4]["sender"]="unknown"
        with self.assertRaisesRegex(ValueError,"Unexpected sender"):
            freeze_rows(a,b)

    def test_csv_columns_cannot_change(self):
        payload=b"source_language,target_language,source\nfr,en,Salut\n"
        with self.assertRaisesRegex(ValueError,"CSV schema"):
            load_csv(payload,SOURCE_FIELDS)

    def test_dropped_gold_reference_is_rejected(self):
        a,b=fake()
        b[0]["reference"]=""
        with self.assertRaisesRegex(ValueError,"empty"):
            freeze_rows(a,b)

    def test_short_test_cannot_be_accepted(self):
        a,b=fake()
        with self.assertRaisesRegex(ValueError,"row count"):
            freeze_rows(a[:-1],b[:-1])

if __name__=="__main__":
    unittest.main()
