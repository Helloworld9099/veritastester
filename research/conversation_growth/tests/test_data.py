"""Offline tests for Tatoeba commercial rights, alignment and review safeguards."""
from __future__ import annotations

import io
import json
import unittest
import zipfile

from research.conversation_growth.data import (
    pairs_from_zip,robust_filter,sample_pairs,settings
)


def opus_zip(pair,records):
    first,second=pair.split("-")
    z=io.BytesIO()
    with zipfile.ZipFile(z,"w") as stream:
        stream.writestr("Tatoeba."+pair+"."+first,"\n".join(x for x,_ in records)+"\n")
        stream.writestr("Tatoeba."+pair+"."+second,"\n".join(y for _,y in records)+"\n")
    return z.getvalue()


class TestAdditionalLicenseCheckedCorpus(unittest.TestCase):
    def test_only_tatoeba_verified_commercial_source_can_train(self):
        cfg=settings()
        self.assertEqual(len(cfg["parallel_sources"]),3)
        self.assertTrue(all(s["license"]=="CC-BY-2.0-FR"
                            for s in cfg["parallel_sources"]))
        self.assertTrue(all("review" in s["use"] for s in cfg["nonparallel_sources"]
                            if "OpenAssistant" in s["id"]))
        self.assertFalse(cfg["final_holdout_access"])
        self.assertFalse(cfg["reviewed_500_as_training"])
        self.assertEqual(cfg["parent"]["selected_adapter_sha256"],
                         "a00c7965211853df07ce3fa032a76283dfc984dcdaeac85e03263445029ffb8e")

    def test_aligned_parallel_zip_and_bad_columns_rejected(self):
        pair="en-es"
        rows=[("Can you bring the blue umbrella now?","¿Puedes traer el paraguas azul ahora?"),
              ("Please close that door before midnight.","Cierra esa puerta antes de medianoche.")]
        self.assertEqual(pairs_from_zip(opus_zip(pair,rows),pair),rows)
        with self.assertRaises(ValueError):
            pairs_from_zip(opus_zip(pair,rows),"en-de")
        with self.assertRaises(zipfile.BadZipFile):
            pairs_from_zip(b"not an archive",pair)

    def test_dedup_numerical_entities_and_private_urls(self):
        raw=[
            ("Can you come tomorrow please?","¿Puedes venir mañana por favor?"),
            ("Can you come tomorrow please?","¿Puedes venir mañana por favor?"),
            ("I will bring 15 bags today","Voy a traer 50 bolsas hoy"),
            ("Email private@example.com immediately","Envía a private@example.com ahora"),
            ("We need to confirm your train booking","Necesitamos confirmar tu reserva de tren"),
            ("The meeting is at 10 PM on Tuesday","La reunión es a las 10 PM el martes")
        ]
        accepted=sample_pairs(raw,"en-es",2,
                              ban={"can you come tomorrow please?"},seed=31)
        self.assertEqual(len(accepted),2)
        self.assertTrue(all(r["license"]=="CC-BY-2.0-FR" for r in accepted))
        self.assertTrue(all("Tatoeba" in r["provenance"] for r in accepted))
        self.assertFalse(robust_filter("Email private@example.com immediately",
                                       "Envía a private@example.com ahora"))
        self.assertFalse(robust_filter("I will bring 15 bags today",
                                       "Voy a traer 50 bolsas hoy"))

    def test_no_monolingual_source_becomes_parallel_without_verified_target(self):
        with self.assertRaises(ValueError):
            sample_pairs([("Hi there","Hola")],"en-de",1,set(),1)
        self.assertFalse(robust_filter("hello","hola"))


if __name__=="__main__":
    unittest.main()
