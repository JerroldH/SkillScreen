"""Pinned recovery checks use synthetic archives, never execute their contents."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.community_data import materialize_inputs
from scripts.restore_community import restore, main


class RecoveryTests(unittest.TestCase):
    def test_existing_partial_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/'inputs.jsonl'
            partial=output.with_suffix('.jsonl.part');partial.write_text('existing work',encoding='utf-8')
            with patch('sys.argv',['restore','--mode','markdown','--offline','--output',str(output)]):
                with self.assertRaises(FileExistsError):main()
            self.assertEqual(partial.read_text(encoding='utf-8'),'existing work')

    def test_full_inventory_hashes_and_matched_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);cache=root/'cache';cache.mkdir()
            markdown='Fixture instructions.\n'
            files={'SKILL.md':b'\xef\xbb\xbfFixture instructions.\r\n',
                   'helper.py':b'print("fixture, never executed")\n',
                   'metadata.json':b'{"label":"malicious"}'}
            archive=root/'fixture.zip'
            with zipfile.ZipFile(archive,'w') as z:
                for name,raw in files.items():z.writestr('case/'+name,raw)
            digest=lambda raw:hashlib.sha256(raw).hexdigest()
            mdhash=digest(markdown.encode()); archivehash=digest(archive.read_bytes())
            (cache/(mdhash+'.blob')).write_bytes(markdown.encode())
            (cache/(archivehash+'.blob')).write_bytes(archive.read_bytes())
            sources={'fixture':{'url':'https://raw.githubusercontent.com/example/repo/revision/fixture.zip','sha256':archivehash,'format':'zip'}}
            row={'id':'fixture','label':'benign','coverage':'package_complete',
                 'recovery':{'markdown':{'url':'https://raw.githubusercontent.com/example/repo/revision/SKILL.md','sha256':mdhash},
                             'package':{'asset':'fixture','prefix':'case/'}},'files':[]}
            for name,raw in files.items():
                row['files'].append({'path':name,'source_path':'case/'+name,'source_sha256':digest(raw),
                                     'sha256':mdhash if name=='SKILL.md' else digest(raw),'bytes':len(raw),
                                     'input_eligible':name!='metadata.json','text':None})
            result=list(restore([row],sources,'package',cache,True))
            self.assertEqual(result[0]['skill_text'],markdown)
            self.assertEqual([f['path'] for f in result[0]['files']],['helper.py'])
            path=root/'inputs.jsonl';path.write_text(json.dumps(result[0])+'\n',encoding='utf-8')
            self.assertEqual(materialize_inputs([row],'package',text_files=[path]),result)
            md=materialize_inputs([row],'markdown',text_files=[path])
            self.assertEqual(md[0]['id'],result[0]['id']);self.assertEqual(md[0]['files'],[])
            missing=copy.deepcopy(row);missing['files'].pop()
            with self.assertRaisesRegex(ValueError,'inventory mismatch'):
                list(restore([missing],sources,'package',cache,True))
            broken=copy.deepcopy(row);broken['files'][1]['source_sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'file mismatch'):
                list(restore([broken],sources,'package',cache,True))
            (cache/(archivehash+'.blob')).write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'Cached source hash mismatch'):
                list(restore([row],sources,'package',cache,True))


if __name__=='__main__':unittest.main()
