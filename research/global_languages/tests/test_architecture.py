"""Unit-level verification that catalog size does not imply translation quality."""
from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
import zipfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

from research.global_languages.catalog import (
    DocumentedCatalog,from_glottolog_csv,parse_iana_subtags,parse_bcp47
)
from research.global_languages.capabilities import (
    GlobalLanguageRouter,native_m2m100_codes,UnsupportedLanguageRoute
)
from research.global_languages.sync_catalog import build_catalog


def iana_fixture():
    records=["File-Date: 2026-11-21"]
    for lang in ("en","es","fr","zh","de","iw","he"):
        records += ["%%","Type: language","Subtag: "+lang,
                    "Description: "+lang.upper()]
        if lang=="iw":
            records.append("Preferred-Value: he")
    for n in range(110):
        code=chr(97+n//26)+chr(97+n%26)+"x"
        records+=["%%","Type: language","Subtag: "+code,
                  "Description: Invented example "+code]
    records+=["%%"]
    return "\n".join(records).encode()


def glotto_fixture():
    out=io.StringIO()
    writer=csv.DictWriter(out,fieldnames=["glottocode","name","level",
                                          "iso639P3code","parent_id"])
    writer.writeheader()
    for row in [
        ("engl1269","English","language","eng",""),
        ("stan1288","Spanish","language","spa",""),
        ("fren1242","French","language","fra",""),
        ("lowd1234","Lower dialect","dialect","","engl1269"),
        ("rare1234","Rare documented language","language","abc",""),
        ("faml1234","Family not language","family","",""),
    ]:
        writer.writerow(dict(zip(writer.fieldnames,row)))
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,"w",compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("glottolog_languoid.csv",out.getvalue())
    return buffer.getvalue()


class FakeTokenizer:
    lang_code_to_id={
        **{"en":1,"es":2,"fr":3,"zh":4,"de":5},
        **{chr(97+i//26)+chr(97+i%26):i+20 for i in range(95)},
    }


class TestUniversalCatalog(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.location=Path(self.temp.name)
        self.report=build_catalog(glotto_fixture(),iana_fixture(),out=self.location)
        self.catalog=DocumentedCatalog.load(self.location)

    def test_glottolog_records_and_license_not_translation_supported(self):
        self.assertEqual(self.report["source_license"],"CC-BY-4.0")
        self.assertEqual(self.report["documented_languages"],4)
        self.assertEqual(self.report["documented_dialects"],1)
        self.assertFalse(self.report["translation_production_supported"])
        self.assertFalse(self.report["meaning_accuracy_verified"])
        self.assertTrue(self.catalog.cataloged("glotto:rare1234"))
        self.assertTrue(self.catalog.cataloged("glotto:lowd1234"))
        self.assertFalse(self.catalog.cataloged("glotto:faml1234"))
        self.assertEqual(self.catalog.lookup("glotto:engl1269").iso639_3,"eng")
        self.assertEqual(self.catalog.lookup("spa").name,"Spanish")
        self.assertIsNone(self.catalog.lookup("zzz"))
        self.assertEqual(self.catalog.stats()["documented_languages"],4)
        self.assertEqual(self.catalog.stats()["translation_ready"],0)

    def test_checksums_detect_modified_catalog_and_iana(self):
        path=self.location/"iana_languages.json"
        path.write_text(path.read_text()+"\n")
        with self.assertRaisesRegex(RuntimeError,"checksum"):
            DocumentedCatalog.load(self.location)

    def test_bcp47_preserves_regional_or_script_diversity(self):
        tag=parse_bcp47("zh-Hant-HK")
        self.assertEqual((tag.primary,tag.script,tag.region),("zh","Hant","HK"))
        self.assertEqual(tag.canonical_tag,"zh-Hant-HK")
        self.assertEqual(parse_bcp47("es-mx").canonical_tag,"es-MX")
        for bad in ("auto","", "x-english", "en-u-ca-gregory",
                    "fr-CA-CA","es-!!","es--MX"):
            with self.subTest(tag=bad),self.assertRaises(ValueError):
                parse_bcp47(bad)

    def test_native_tokenizer_not_entire_glottolog_inventory(self):
        router=GlobalLanguageRouter(self.catalog,FakeTokenizer())
        supported=router.inspect("es","fr")
        self.assertTrue(supported.can_run_native_model)
        self.assertTrue(supported.research_finetuned)
        self.assertFalse(supported.independently_meaning_verified)
        self.assertFalse(supported.production_approved)
        self.assertEqual(supported.model_coverage,"research_finetuned_unvalidated")
        region=router.inspect("es-MX","fr-CA")
        self.assertTrue(region.can_run_native_model)
        self.assertEqual(region.model_coverage,"native_variant_unvalidated")
        self.assertFalse(region.regional_variant_evaluated)
        self.assertEqual(router.inspect("zh-Hant-HK","en").model_coverage,
                         "native_variant_unvalidated")
        catalog_only=router.inspect("glotto:rare1234","en")
        self.assertTrue(catalog_only.source_documented)
        self.assertFalse(catalog_only.can_run_native_model)
        self.assertEqual(catalog_only.model_coverage,"catalog_only")
        with self.assertRaises(UnsupportedLanguageRoute):
            router.require_research_route("glotto:rare1234","en")
        self.assertEqual(router.require_research_route("glotto:engl1269","fr"),
                         ("en","fr"))
        self.assertEqual(router.inspect("zzzzz","en").model_coverage,
                         "unknown_identifier")

    def test_no_fake_model_support_for_catalog_only_language(self):
        router=GlobalLanguageRouter(self.catalog,FakeTokenizer())
        info=router.coverage_stats()
        self.assertGreaterEqual(info["native_m2m100_language_codes"],90)
        self.assertEqual(info["independently_human_validated_pairs"],0)
        self.assertEqual(info["production_multilingual_routes_activated"],0)
        self.assertEqual(info["native_language_pairs_theoretically_possible"],
                         info["native_m2m100_language_codes"]*
                         (info["native_m2m100_language_codes"]-1))
        with self.assertRaises(UnsupportedLanguageRoute):
            router.require_research_route("glotto:lowd1234","es")

    def test_false_zip_cannot_write_or_claim_unsupported_catalog(self):
        with self.assertRaises(zipfile.BadZipFile):
            build_catalog(b"not a zip",iana_fixture(),out=self.location/"bad")
        with tempfile.TemporaryDirectory() as tmp:
            blank=Path(tmp)/"empty"
            with self.assertRaises(ValueError):
                build_catalog(glotto_fixture(),b"no official IANA data",out=blank)


if __name__=="__main__":
    unittest.main()
