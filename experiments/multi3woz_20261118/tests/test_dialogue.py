"""Unit tests for Multi3WOZ dialogue split firewall and reject obvious bad translations."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train_dialogue import eligible_pair, choose_dialogue_cohorts


def turn(source, english, idx):
    return {"text":source,"en_text":english,"speaker":"USER",
            "intent_with_errors":[],"turn_id":idx}


def dialogue(label, count):
    items=[]
    for i in range(count):
        items.append(turn(
            f"Bonjour je voudrais réserver une chambre dans quartier {label} numero {i}",
            f"Hello I would like to book a room in district {label} number {i}",i))
    return {"goal":{},"log":items}


class TestMulti3WOZ(unittest.TestCase):
    def test_culturally_expanded_reference_is_not_safe_alignment(self):
        original=turn("Bonjour, Je cherche un restaurant indien.",
                      "I'd really like to take my client out to a nice restaurant that serves indian food .",0)
        self.assertIsNone(eligible_pair(original))

    def test_number_mismatch_and_error_labels_rejected(self):
        a=turn("Je voudrais un restaurant pour 3 personnes maintenant",
               "I would like a restaurant for 4 people now",0)
        self.assertIsNone(eligible_pair(a))
        a["en_text"]="I would like a restaurant for 3 people now"
        self.assertIsNotNone(eligible_pair(a))
        a["intent_with_errors"]=["Restaurant-Inform"]
        self.assertIsNone(eligible_pair(a))

    def test_disjoint_dialogue_ids_and_exact_source_rows(self):
        rows={}
        for group in ("train","valid","test"):
            for j in range(5):
                key=f"{group}-{j}.json"
                rows[key]=dialogue(key,4)
        validation={f"valid-{j}.json" for j in range(5)}
        test={f"test-{j}.json" for j in range(5)}
        split, hashes=choose_dialogue_cohorts(rows,validation,test,
                                             train_n=5,dev_n=5,test_n=5,seed=18)
        self.assertEqual([len(split[k]) for k in ("train","dev","test")],[5,5,5])
        self.assertEqual(len(hashes["test"]["dialogue_ids_sha256"]),64)
        self.assertEqual(split,choose_dialogue_cohorts(
            rows,validation,test,train_n=5,dev_n=5,test_n=5,seed=18)[0])
        self.assertFalse(set(x[0] for x in split["test"]) &
                         set(x[0] for x in split["train"]))

    def test_duplicate_sentences_do_not_enter_multiple_partitions(self):
        rows={}
        for group in ("train","valid","test"):
            rows[group+".json"]=dialogue("same-label",3)
        with self.assertRaisesRegex(ValueError,"filtered"):
            choose_dialogue_cohorts(rows,{"valid.json"},{"test.json"},
                                    train_n=2,dev_n=2,test_n=2)

    def test_official_test_never_becomes_train(self):
        rows={"train.json":dialogue("train",3),
              "dev.json":dialogue("dev",3),
              "test.json":dialogue("test",3)}
        with self.assertRaisesRegex(ValueError,"Missing/overlapping"):
            choose_dialogue_cohorts(rows,{"dev.json"},{"dev.json"},
                                    train_n=1,dev_n=1,test_n=1)


if __name__=="__main__":
    unittest.main()
