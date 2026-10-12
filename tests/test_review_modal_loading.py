"""Exercise asynchronous modal loading without a browser dependency."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


def test_lazy_modal_deduplicates_retries_and_checks_authentication() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    script_path = Path(__file__).resolve().parents[1] / "assets/js/review_modal.js"
    script = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const modals = new Map();
const listeners = new Map();
let reloads = 0;
let requests = 0, shown = 0, redirects = 0, errors = 0;
let response = {ok:true, status:200, redirected:false, text:async()=>'<div>modal</div>'};
const tile = {
    dataset:{modalTarget:'#modal-review-test-1', modalUrl:'/api/review/modal/test/1'},
    isConnected:true,
    closest:()=>({querySelector:()=>tile}),
    setAttribute:()=>{}, removeAttribute:()=>{}
};
const context = {
    window:{wmToast:()=>errors++, location:{reload:()=>reloads++}},
    document:{
        addEventListener:(name, callback)=>listeners.set(name, callback),
        querySelector:key=>modals.get(key),
        createElement:()=>({content:{querySelector:()=>({id:'modal-review-test-1',dataset:{}})}}),
        body:{appendChild:modal=>modals.set('#'+modal.id,modal)}
    },
    fetch:async()=>{requests++;return response;},
    safeSameOriginPath:url=>url,
    redirectToLogin:()=>redirects++,
    bootstrap:{Modal:{getOrCreateInstance:()=>({show:()=>shown++})}},
    requestAnimationFrame:callback=>callback(),
    modalNavigationInFlight:false,
    Map
};
vm.createContext(context);
vm.runInContext(fs.readFileSync(SCRIPT_PATH,'utf8'),context);
const open = context.window.WmReviewModal.open;
(async()=>{
    // A successful HTML fragment is not a login response.
    await Promise.all([open(tile),open(tile)]);
    assert.equal(requests,1);
    assert.equal(shown,1);
    assert.equal(redirects,0);
    await open(tile);
    assert.equal(requests,1);
    assert.equal(modals.size,1);

    modals.clear();
    response={ok:false,status:500,redirected:false};
    await open(tile);
    assert.equal(errors,1);
    assert.equal(modals.size,0);
    response={ok:true,status:200,redirected:false,text:async()=>'<div>modal</div>'};
    await open(tile);
    assert.equal(requests,3);
    assert.equal(modals.size,1);

    modals.clear();
    response={ok:true,status:200,redirected:true};
    await open(tile);
    assert.equal(redirects,1);
    assert.equal(modals.size,0);

    response={ok:false,status:404,redirected:false};
    await open(tile);
    assert.equal(modals.size,0);
    assert.equal(errors,3);

    listeners.get('hidden.bs.modal')();
    assert.equal(reloads,0);
    listeners.get('wmb:bird-objects-updated')();
    context.modalNavigationInFlight = true;
    listeners.get('hidden.bs.modal')();
    assert.equal(reloads,0);
    context.modalNavigationInFlight = false;
    listeners.get('hidden.bs.modal')();
    assert.equal(reloads,1);
})().catch(error=>{console.error(error);process.exitCode=1;});
""".replace("SCRIPT_PATH", json.dumps(str(script_path)))
    subprocess.run([node, "-e", script], check=True, capture_output=True, text=True)
