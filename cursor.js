(function () {
  'use strict';

  if (window.matchMedia('(pointer: coarse)').matches) return;
  var THREE = window.THREE;
  if (!THREE) return;

  // ── Canvas ────────────────────────────────────────────────────────────────
  var canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;inset:0;width:100%;height:100%;pointer-events:none;z-index:9999;mix-blend-mode:difference;';
  document.body.appendChild(canvas);

  var renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, premultipliedAlpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.autoClear = false;
  renderer.setClearColor(0x000000, 0);

  // ── Scene / Quad ──────────────────────────────────────────────────────────
  var scene  = new THREE.Scene();
  var camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

  var geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(
      [-1,-1,0, 1,-1,0, -1,1,0, 1,-1,0, 1,1,0, -1,1,0], 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(
      [0,0, 1,0, 0,1, 1,0, 1,1, 0,1], 2));
  var quad = new THREE.Mesh(geo);
  scene.add(quad);

  // ── Config ────────────────────────────────────────────────────────────────
  var SIM_RES   = 256;
  var DYE_RES   = 1024;
  var CURL      = 30;
  var PRESSURE_ITER = 20;
  var VEL_DISS  = 0.98;
  var DYE_DISS  = 0.97;
  var SPLAT_R   = 0.25;   // % of viewport (divided by 100 for shader)
  var FORCE     = 6000;
  var P_DECAY   = 0.8;
  var THRESHOLD = 0.3;
  var SOFTNESS  = 0.0;

  // ── FBO helpers ───────────────────────────────────────────────────────────
  function fboSize(res) {
    var ar = window.innerWidth / window.innerHeight;
    return ar > 1
      ? { w: res, h: Math.round(res / ar) }
      : { w: Math.round(res * ar), h: res };
  }

  function mkFBO(w, h, filter) {
    return new THREE.WebGLRenderTarget(w, h, {
      type:         THREE.HalfFloatType,
      format:       THREE.RGBAFormat,
      minFilter:    filter || THREE.LinearFilter,
      magFilter:    filter || THREE.LinearFilter,
      wrapS:        THREE.ClampToEdgeWrapping,
      wrapT:        THREE.ClampToEdgeWrapping,
      depthBuffer:  false,
      stencilBuffer:false,
    });
  }

  function mkDouble(w, h, filter) {
    var a = mkFBO(w, h, filter), b = mkFBO(w, h, filter);
    return { read: a, write: b, swap: function(){ var t=this.read; this.read=this.write; this.write=t; } };
  }

  var ss = fboSize(SIM_RES);
  var ds = fboSize(DYE_RES);
  var simTS = new THREE.Vector2(1/ss.w, 1/ss.h);
  var dyeTS = new THREE.Vector2(1/ds.w, 1/ds.h);

  var vel      = mkDouble(ss.w, ss.h);
  var dye      = mkDouble(ds.w, ds.h);
  var divFBO   = mkFBO(ss.w, ss.h, THREE.NearestFilter);
  var curlFBO  = mkFBO(ss.w, ss.h, THREE.NearestFilter);
  var pres     = mkDouble(ss.w, ss.h, THREE.NearestFilter);

  // ── Vertex shader (shared) ────────────────────────────────────────────────
  var VS = [
    'precision highp float;',
    'attribute vec3 position;',
    'attribute vec2 uv;',
    'varying vec2 vUv;',
    'void main(){vUv=uv;gl_Position=vec4(position.xy,0.,1.);}',
  ].join('\n');

  function mat(fs, u) {
    return new THREE.RawShaderMaterial({
      vertexShader: VS, fragmentShader: fs, uniforms: u||{},
      depthTest: false, depthWrite: false,
    });
  }

  // ── Fragment shaders ──────────────────────────────────────────────────────
  var clearMat = mat([
    'precision mediump float;',
    'uniform sampler2D uTex; uniform float value; varying vec2 vUv;',
    'void main(){gl_FragColor=value*texture2D(uTex,vUv);}',
  ].join('\n'), { uTex:{value:null}, value:{value:P_DECAY} });

  var splatMat = mat([
    'precision highp float;',
    'uniform sampler2D uTarget; uniform float ar; uniform vec3 color;',
    'uniform vec2 point; uniform float radius; varying vec2 vUv;',
    'void main(){',
      'vec2 p=vUv-point; p.x*=ar;',
      'vec3 s=exp(-dot(p,p)/radius)*color;',
      'gl_FragColor=vec4(texture2D(uTarget,vUv).xyz+s,1.);',
    '}',
  ].join('\n'), {
    uTarget:{value:null}, ar:{value:1}, color:{value:new THREE.Vector3()},
    point:{value:new THREE.Vector2()}, radius:{value:0},
  });

  var advectMat = mat([
    'precision highp float;',
    'uniform sampler2D uVel; uniform sampler2D uSrc;',
    'uniform vec2 simTS; uniform vec2 srcTS; uniform float dt; uniform float diss;',
    'varying vec2 vUv;',
    'vec4 bilerp(sampler2D s,vec2 uv,vec2 ts){',
      'vec2 st=uv/ts-0.5; vec2 f=fract(st); vec2 i=floor(st);',
      'vec4 a=texture2D(s,(i+vec2(.5,.5))*ts); vec4 b=texture2D(s,(i+vec2(1.5,.5))*ts);',
      'vec4 c=texture2D(s,(i+vec2(.5,1.5))*ts); vec4 d=texture2D(s,(i+vec2(1.5,1.5))*ts);',
      'return mix(mix(a,b,f.x),mix(c,d,f.x),f.y);',
    '}',
    'void main(){',
      'vec2 coord=vUv-dt*bilerp(uVel,vUv,simTS).xy*simTS;',
      'gl_FragColor=diss*bilerp(uSrc,coord,srcTS);',
    '}',
  ].join('\n'), {
    uVel:{value:null}, uSrc:{value:null},
    simTS:{value:simTS.clone()}, srcTS:{value:simTS.clone()},
    dt:{value:0.016}, diss:{value:1.0},
  });

  var divMat = mat([
    'precision mediump float;',
    'uniform sampler2D uVel; uniform vec2 ts; varying vec2 vUv;',
    'void main(){',
      'float L=texture2D(uVel,vUv-vec2(ts.x,0.)).x;',
      'float R=texture2D(uVel,vUv+vec2(ts.x,0.)).x;',
      'float T=texture2D(uVel,vUv+vec2(0.,ts.y)).y;',
      'float B=texture2D(uVel,vUv-vec2(0.,ts.y)).y;',
      'gl_FragColor=vec4(.5*(R-L+T-B),0.,0.,1.);',
    '}',
  ].join('\n'), { uVel:{value:null}, ts:{value:simTS.clone()} });

  var curlMat = mat([
    'precision mediump float;',
    'uniform sampler2D uVel; uniform vec2 ts; varying vec2 vUv;',
    'void main(){',
      'float L=texture2D(uVel,vUv-vec2(ts.x,0.)).y;',
      'float R=texture2D(uVel,vUv+vec2(ts.x,0.)).y;',
      'float T=texture2D(uVel,vUv+vec2(0.,ts.y)).x;',
      'float B=texture2D(uVel,vUv-vec2(0.,ts.y)).x;',
      'gl_FragColor=vec4(.5*(R-L-T+B),0.,0.,1.);',
    '}',
  ].join('\n'), { uVel:{value:null}, ts:{value:simTS.clone()} });

  var vortMat = mat([
    'precision highp float;',
    'uniform sampler2D uVel; uniform sampler2D uCurl;',
    'uniform vec2 ts; uniform float curl; uniform float dt; varying vec2 vUv;',
    'void main(){',
      'float L=texture2D(uCurl,vUv-vec2(ts.x,0.)).x;',
      'float R=texture2D(uCurl,vUv+vec2(ts.x,0.)).x;',
      'float T=texture2D(uCurl,vUv+vec2(0.,ts.y)).x;',
      'float B=texture2D(uCurl,vUv-vec2(0.,ts.y)).x;',
      'float C=texture2D(uCurl,vUv).x;',
      'vec2 f=.5*vec2(abs(T)-abs(B),abs(R)-abs(L));',
      'f/=length(f)+.0001; f*=curl*C; f.y*=-1.;',
      'gl_FragColor=vec4(texture2D(uVel,vUv).xy+f*dt,0.,1.);',
    '}',
  ].join('\n'), {
    uVel:{value:null}, uCurl:{value:null},
    ts:{value:simTS.clone()}, curl:{value:CURL}, dt:{value:0.016},
  });

  var presMat = mat([
    'precision mediump float;',
    'uniform sampler2D uPres; uniform sampler2D uDiv; uniform vec2 ts; varying vec2 vUv;',
    'void main(){',
      'float L=texture2D(uPres,vUv-vec2(ts.x,0.)).x;',
      'float R=texture2D(uPres,vUv+vec2(ts.x,0.)).x;',
      'float T=texture2D(uPres,vUv+vec2(0.,ts.y)).x;',
      'float B=texture2D(uPres,vUv-vec2(0.,ts.y)).x;',
      'float d=texture2D(uDiv,vUv).x;',
      'gl_FragColor=vec4((L+R+B+T-d)*.25,0.,0.,1.);',
    '}',
  ].join('\n'), { uPres:{value:null}, uDiv:{value:null}, ts:{value:simTS.clone()} });

  var gradMat = mat([
    'precision mediump float;',
    'uniform sampler2D uPres; uniform sampler2D uVel; uniform vec2 ts; varying vec2 vUv;',
    'void main(){',
      'float L=texture2D(uPres,vUv-vec2(ts.x,0.)).x;',
      'float R=texture2D(uPres,vUv+vec2(ts.x,0.)).x;',
      'float T=texture2D(uPres,vUv+vec2(0.,ts.y)).x;',
      'float B=texture2D(uPres,vUv-vec2(0.,ts.y)).x;',
      'vec2 v=texture2D(uVel,vUv).xy;',
      'gl_FragColor=vec4(v-vec2(R-L,T-B),0.,1.);',
    '}',
  ].join('\n'), { uPres:{value:null}, uVel:{value:null}, ts:{value:simTS.clone()} });

  var dispMat = mat([
    'precision highp float;',
    'uniform sampler2D uTex; uniform float threshold; uniform float softness; varying vec2 vUv;',
    'void main(){',
      'vec3 c=texture2D(uTex,vUv).rgb;',
      'float b=max(c.r,max(c.g,c.b));',
      'float a=smoothstep(threshold-max(softness,.001),threshold+max(softness,.001),b);',
      'gl_FragColor=vec4(1.,1.,1.,a);',
    '}',
  ].join('\n'), {
    uTex:{value:null}, threshold:{value:THRESHOLD}, softness:{value:SOFTNESS},
  });

  // ── Pass helper ───────────────────────────────────────────────────────────
  function pass(m, target) {
    quad.material = m;
    renderer.setRenderTarget(target || null);
    renderer.render(scene, camera);
  }

  // ── Splat ─────────────────────────────────────────────────────────────────
  function splat(x, y, dx, dy) {
    var ar = window.innerWidth / window.innerHeight;
    var r  = (ar > 1 ? SPLAT_R * ar : SPLAT_R) / 100;

    splatMat.uniforms.ar.value = ar;
    splatMat.uniforms.point.value.set(x, y);
    splatMat.uniforms.radius.value = r;

    splatMat.uniforms.uTarget.value = vel.read.texture;
    splatMat.uniforms.color.value.set(dx, dy, 0);
    pass(splatMat, vel.write);
    vel.swap();

    splatMat.uniforms.uTarget.value = dye.read.texture;
    splatMat.uniforms.color.value.set(1, 1, 1);
    pass(splatMat, dye.write);
    dye.swap();
  }

  // ── Simulate ──────────────────────────────────────────────────────────────
  function simulate(dt) {
    curlMat.uniforms.uVel.value = vel.read.texture;
    pass(curlMat, curlFBO);

    vortMat.uniforms.uVel.value  = vel.read.texture;
    vortMat.uniforms.uCurl.value = curlFBO.texture;
    vortMat.uniforms.dt.value    = dt;
    pass(vortMat, vel.write); vel.swap();

    divMat.uniforms.uVel.value = vel.read.texture;
    pass(divMat, divFBO);

    clearMat.uniforms.uTex.value = pres.read.texture;
    pass(clearMat, pres.write); pres.swap();

    for (var i = 0; i < PRESSURE_ITER; i++) {
      presMat.uniforms.uPres.value = pres.read.texture;
      presMat.uniforms.uDiv.value  = divFBO.texture;
      pass(presMat, pres.write); pres.swap();
    }

    gradMat.uniforms.uPres.value = pres.read.texture;
    gradMat.uniforms.uVel.value  = vel.read.texture;
    pass(gradMat, vel.write); vel.swap();

    // advect velocity
    advectMat.uniforms.uVel.value  = vel.read.texture;
    advectMat.uniforms.uSrc.value  = vel.read.texture;
    advectMat.uniforms.simTS.value.copy(simTS);
    advectMat.uniforms.srcTS.value.copy(simTS);
    advectMat.uniforms.dt.value    = dt;
    advectMat.uniforms.diss.value  = VEL_DISS;
    pass(advectMat, vel.write); vel.swap();

    // advect dye
    advectMat.uniforms.uVel.value  = vel.read.texture;
    advectMat.uniforms.uSrc.value  = dye.read.texture;
    advectMat.uniforms.srcTS.value.copy(dyeTS);
    advectMat.uniforms.diss.value  = DYE_DISS;
    pass(advectMat, dye.write); dye.swap();
  }

  // ── Render to screen ──────────────────────────────────────────────────────
  function render() {
    renderer.setRenderTarget(null);
    renderer.clear();
    dispMat.uniforms.uTex.value = dye.read.texture;
    pass(dispMat, null);
  }

  // ── Input ─────────────────────────────────────────────────────────────────
  var mx = 0.5, my = 0.5, px = 0.5, py = 0.5, moved = false;

  window.addEventListener('mousemove', function (e) {
    px = mx; py = my;
    mx = e.clientX / window.innerWidth;
    my = 1 - e.clientY / window.innerHeight;
    moved = true;
  });

  window.addEventListener('resize', function () {
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  // ── Loop ──────────────────────────────────────────────────────────────────
  var last = performance.now();

  function tick() {
    requestAnimationFrame(tick);
    var now = performance.now();
    var dt  = Math.min((now - last) / 1000, 0.016);
    last = now;

    if (moved) {
      splat(mx, my, (mx - px) * FORCE, (my - py) * FORCE);
      moved = false;
    }

    simulate(dt);
    render();
  }

  requestAnimationFrame(tick);
}());
