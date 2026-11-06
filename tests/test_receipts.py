import json
import unittest
from common.demo import Receipts


class Page:
    def on(self,*args): pass


class Request:
    def __init__(self,url): self.url=url


class Response:
    def __init__(self,request,body,status=200):
        self.request,self.body,self.status,self.url=request,body,status,request.url
    async def text(self): return json.dumps(self.body)


class ReceiptTests(unittest.IsolatedAsyncioTestCase):
    async def check(self,site,url,body,token=None,status=200):
        receipts=Receipts(Page(),site)
        receipts.active="attempt-1";receipts.token=token
        request=Request(url);receipts.request(request)
        # 回执即使晚到也只能属于原请求的尝试。
        receipts.active="attempt-2"
        await receipts.response(Response(request,body,status))
        return receipts.results

    async def test_kg_success_requires_token(self):
        results=await self.check("kg","https://cdn9.kgcaptcha.com/requestFront",{"code":0})
        self.assertIsNone(results[0]['accepted'])

    async def test_kg_wrong_answer_http_200_is_failed(self):
        results=await self.check("kg","https://cdn9.kgcaptcha.com/requestFront",{"code":1,"msg":"FAIL"})
        self.assertIs(results[0]['accepted'],False)

    async def test_response_keeps_original_attempt(self):
        results=await self.check("kg","https://cdn9.kgcaptcha.com/requestFront",{"code":0,"token":"fixture"})
        self.assertEqual(results[0]['attempt_id'],"attempt-1")
        self.assertIs(results[0]['accepted'],True)

    async def test_unrelated_host_is_ignored(self):
        results=await self.check("kg","https://unrelated.invalid/requestFront",{"code":0,"token":"fixture"})
        self.assertEqual(results,[])

    async def test_unrelated_generation_cannot_change_current_token(self):
        receipts=Receipts(Page(),'yidun');receipts.token='original'
        req=Request('https://unrelated.invalid/api/v3/get')
        await receipts.response(Response(req,{'data':{'token':'unrelated'}}))
        self.assertEqual(receipts.token,'original')

    async def test_parse_error_is_bound_to_original_request(self):
        receipts=Receipts(Page(),'kg');receipts.active='original'
        req=Request('https://cdn9.kgcaptcha.com/requestFront');receipts.request(req)
        receipts.active='later'
        await receipts.response(Response(req,['unexpected-array']))
        self.assertEqual(receipts.results[0]['attempt_id'],'original')
        self.assertIsNone(receipts.results[0]['accepted'])

    async def test_yidun_conflicting_token_cannot_pass(self):
        results=await self.check("yidun","https://c.dun.163.com/api/v3/check?token=A",
            {"error":0,"data":{"result":True,"token":"B","validate":"fixture"}},token="A")
        self.assertIsNone(results[0]['accepted'])

    async def test_yidun_validated_current_challenge(self):
        results=await self.check("yidun","https://c.dun.163.com/api/v3/check?token=A",
            {"error":0,"data":{"result":True,"token":"A","validate":"fixture"}},token="A")
        self.assertIs(results[0]['accepted'],True)

    async def test_new_generation_during_action_cannot_pass_old_attempt(self):
        receipts=Receipts(Page(),'yidun');receipts.active='old-attempt'
        receipts.expected_token='old';receipts.token='new'
        req=Request('https://c.dun.163.com/api/v3/check?token=new');receipts.request(req)
        await receipts.response(Response(req,{'error':0,'data':{'result':True,'token':'new','validate':'fixture'}}))
        self.assertIsNone(receipts.results[0]['accepted'])

    async def test_http_error_never_passes(self):
        results=await self.check("kg","https://cdn9.kgcaptcha.com/requestFront",{"code":0,"token":"fixture"},status=500)
        self.assertIsNone(results[0]['accepted'])

    async def test_challenge_generation_requires_valid_response(self):
        receipts=Receipts(Page(),'yidun')
        req=Request('https://c.dun.163.com/api/v3/get')
        await receipts.response(Response(req,{'error':1,'data':{'token':'bad'}}))
        self.assertEqual(receipts.generation,0)
        for token in ['first','second']:
            await receipts.response(Response(req,{'error':0,'data':{'token':token}}))
        self.assertEqual(receipts.generation,2)
        self.assertEqual(receipts.token,'second')
        self.assertNotIn('second',json.dumps(receipts.generations))


if __name__=="__main__": unittest.main()
