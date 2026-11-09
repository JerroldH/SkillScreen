"""Exercise conflicting package variants and inert-text input filtering."""
import copy
import hashlib
import unittest

from scripts.attach_package_views import attach
from scripts.import_malskillbench import normalized_hash
from scripts.recover_benign_packages import decode_input


def fixture():
    text = 'Fixture Markdown'
    key = normalized_hash(text)
    pool = [{'id':key,'label':0,'skill_text':text,'normalized_sha256':key,
             'members':[{'id':'source-a'},{'id':'source-b'}]}]
    row = {'source_member_id':'source-a','label':0,'skill_text':text,
           'files':[{'path':'scripts/helper.py','text':'Fixture helper'}],
           'inventory':[{'path':'SKILL.md','sha256':hashlib.sha256(text.encode()).hexdigest(),'input_eligible':True},
                        {'path':'scripts/helper.py','sha256':hashlib.sha256(b'Fixture helper').hexdigest(),'input_eligible':True}],
           'package_complete':True,'source_url':'https://example.org/source','revision':'a'*40,
           'source_package_path':'fixture','upstream_license':'test-fixture'}
    return pool,row


class PackageViewTests(unittest.TestCase):
    def test_ambiguous_variants_never_combined_and_priority_is_explicit(self):
        pool,one = fixture()
        two = copy.deepcopy(one);two['source_member_id']='source-b'
        two['files'][0]['text']='Different helper'
        two['inventory'][1]['sha256']=hashlib.sha256(b'Different helper').hexdigest()
        coverage,views,_=attach(pool,[[one,two]])
        self.assertEqual(coverage[0]['coverage'],'package_ambiguous')
        self.assertEqual(views,[])
        coverage,views,_=attach(pool,[[one],[two]])
        self.assertEqual(coverage[0]['source_member_id'],'source-a')
        self.assertEqual(views[0]['files'],one['files'])
        self.assertEqual(views[0]['skill_text'],pool[0]['skill_text'])

    def test_partial_bad_label_and_tampered_companion(self):
        pool,row=fixture();row['package_complete']=False
        coverage,views,_=attach(pool,[[row]])
        self.assertEqual(coverage[0]['coverage'],'package_partial');self.assertEqual(views,[])
        row['label']=1
        with self.assertRaisesRegex(ValueError,'identity mismatch'):attach(pool,[[row]])
        pool,row=fixture();row['files'][0]['text']='Tampered'
        with self.assertRaisesRegex(ValueError,'text/hash mismatch'):attach(pool,[[row]])

    def test_metadata_and_private_configuration_are_not_model_inputs(self):
        for path in ['_meta.json','metadata.json','BENCHMARK_INJECTION.md','.env','secret.key']:
            text,reason=decode_input(path,b'Fixture annotation or private config')
            self.assertIsNone(text);self.assertTrue(reason)
        self.assertEqual(decode_input('scripts/helper.py',b'Fixture inert code'),('Fixture inert code',''))
        self.assertEqual(decode_input('asset.bin',b'\x00'),(None,'not_utf8_text'))
        with self.assertRaises(ValueError):decode_input('../outside.py',b'Fixture')


if __name__=='__main__':unittest.main()
