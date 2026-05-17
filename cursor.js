(function () {
  'use strict';

  if (window.matchMedia('(pointer: coarse)').matches) return;

  var canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;inset:0;width:100%;height:100%;pointer-events:none;z-index:9999;mix-blend-mode:screen;';
  document.body.appendChild(canvas);

  var gl = canvas.getContext('webgl2');
  if (!gl) { canvas.remove(); return; }
  if (!gl.getExtension('EXT_color_buffer_float')) { canvas.remove(); return; }

  var w = 0, h = 0, simW = 0, simH = 0;

  // ── shaders ──────────────────────────────────────────────────────────────

  var VS = '#version 300 es\n' +
    'in vec2 a_pos;out vec2 v_uv;\n' +
    'void main(){gl_Position=vec4(a_pos,0.0,1.0);v_uv=a_pos*0.5+0.5;}';

  // Fluid simulation — faithful port of lusion's frag$n
  var SIM_FS = '#version 300 es\n' +
    'precision highp float;\n' +
    'uniform sampler2D u_prev;\n' +
    'uniform vec2 u_texelSize;\n' +
    'uniform vec4 u_from;\n' +   // xy = pixel pos in sim space, z = radius, w = weight
    'uniform vec4 u_to;\n' +
    'uniform vec2 u_vel;\n' +
    'uniform float u_pushStrength;\n' +
    'uniform vec3 u_dissipations;\n' +  // x=velDiss, y=weight1Diss, z=weight2Diss
    'in vec2 v_uv;out vec4 fragColor;\n' +
    'vec2 sdSegment(vec2 p,vec2 a,vec2 b){' +
      'vec2 pa=p-a,ba=b-a;' +
      'float h=clamp(dot(pa,ba)/dot(ba,ba),0.0,1.0);' +
      'return vec2(length(pa-ba*h),h);' +
    '}' +
    'void main(){' +
      'vec2 res=sdSegment(gl_FragCoord.xy,u_from.xy,u_to.xy);' +
      'vec2 radiusWeight=mix(u_from.zw,u_to.zw,res.y);' +
      'float d=1.0-smoothstep(-0.01,radiusWeight.x,res.x);' +
      // self-advection: look back along flow to find where fluid came from
      'vec4 lowData=texture(u_prev,v_uv);' +
      'vec2 velInv=(0.5-lowData.xy)*u_pushStrength;' +
      'vec4 data=texture(u_prev,v_uv+velInv*u_texelSize);' +
      'data.xy-=0.5;' +
      // dissipation
      'vec4 delta=vec4(u_dissipations.xxy,u_dissipations.z)-1.0;' +
      'delta*=data;' +
      // stamp mouse velocity
      'delta+=vec4(u_vel*d,radiusWeight.yy*d);' +
      'delta.zw=sign(delta.zw)*max(vec2(0.004),abs(delta.zw));' +
      'data+=delta;' +
      'data.xy+=0.5;' +
      'fragColor=clamp(data,vec4(0.0),vec4(1.0));' +
    '}';

  // Display — iridescent colour from velocity field (lusion's sin() term)
  var DISP_FS = '#version 300 es\n' +
    'precision highp float;\n' +
    'uniform sampler2D u_paint;\n' +
    'uniform float u_rgbShift;\n' +
    'uniform float u_colorMultiplier;\n' +
    'uniform float u_shade;\n' +
    'in vec2 v_uv;out vec4 fragColor;\n' +
    'void main(){' +
      'vec4 data=texture(u_paint,v_uv);' +
      'float weight=(data.z+data.w)*0.5;' +
      'vec2 vel=(0.5-data.xy-0.001)*2.0*weight;' +
      // exact formula from lusion's frag$1
      'vec3 col=sin(vec3(vel.x+vel.y)*40.0+vec3(0.0,2.0,4.0)*u_rgbShift)' +
        '*smoothstep(0.4,-0.9,weight)' +
        '*u_shade' +
        '*max(abs(vel.x),abs(vel.y))' +
        '*u_colorMultiplier;' +
      // abs() so colours are always additive with mix-blend-mode:screen
      'col=abs(col);' +
      'float alpha=clamp(weight*3.0,0.0,1.0);' +
      'fragColor=vec4(col,alpha);' +
    '}';

  // ── program helpers ───────────────────────────────────────────────────────

  function makeShader(type, src) {
    var s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    return s;
  }

  function makeProgram(vsSrc, fsSrc) {
    var p = gl.createProgram();
    gl.attachShader(p, makeShader(gl.VERTEX_SHADER, vsSrc));
    gl.attachShader(p, makeShader(gl.FRAGMENT_SHADER, fsSrc));
    gl.linkProgram(p);
    return p;
  }

  var simProg  = makeProgram(VS, SIM_FS);
  var dispProg = makeProgram(VS, DISP_FS);

  // quad
  var quad = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quad);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);

  function bindQuad(prog) {
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    var loc = gl.getAttribLocation(prog, 'a_pos');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  }

  // ── FBO ping-pong ─────────────────────────────────────────────────────────

  var ping = null, pong = null;  // {fbo, tex}
  var curr = 0;                  // which slot is "current write target"

  function makeFBO(fw, fh) {
    var tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, fw, fh, 0, gl.RGBA, gl.HALF_FLOAT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    var fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    return { fbo: fbo, tex: tex };
  }

  // Init pass — fills FBO with (0.5, 0.5, 0, 0) meaning zero velocity
  var initProg = makeProgram(VS,
    '#version 300 es\nprecision highp float;\nin vec2 v_uv;out vec4 fragColor;\n' +
    'void main(){fragColor=vec4(0.5,0.5,0.0,0.0);}'
  );

  function clearFBO(slot) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, slot.fbo);
    gl.viewport(0, 0, simW, simH);
    gl.useProgram(initProg);
    bindQuad(initProg);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  function initFBOs() {
    if (ping) { gl.deleteTexture(ping.tex); gl.deleteFramebuffer(ping.fbo); }
    if (pong) { gl.deleteTexture(pong.tex); gl.deleteFramebuffer(pong.fbo); }
    ping = makeFBO(simW, simH);
    pong = makeFBO(simW, simH);
    clearFBO(ping);
    clearFBO(pong);
    curr = 0;
  }

  function resize() {
    w = window.innerWidth;
    h = window.innerHeight;
    simW = Math.max(1, Math.ceil(w / 4));
    simH = Math.max(1, Math.ceil(h / 4));
    canvas.width  = w;
    canvas.height = h;
    initFBOs();
  }

  window.addEventListener('resize', resize);
  resize();

  // ── mouse tracking ────────────────────────────────────────────────────────

  var mouseX = w / 2, mouseY = h / 2;
  var prevX  = w / 2, prevY  = h / 2;
  var velX = 0, velY = 0;
  var hasMoved = false;

  window.addEventListener('mousemove', function (e) {
    prevX  = mouseX;  prevY  = mouseY;
    mouseX = e.clientX;  mouseY = e.clientY;
    velX   = mouseX - prevX;
    velY   = mouseY - prevY;
    hasMoved = true;
  });

  // ── params (matching lusion defaults) ─────────────────────────────────────

  var RADIUS        = 18;    // sim-space pixels
  var WEIGHT        = 1.0;
  var PUSH_STRENGTH = 10;
  var DISS_VEL      = 0.985;
  var DISS_W1       = 0.985;
  var DISS_W2       = 0.5;
  var RGB_SHIFT     = 1.0;
  var COLOR_MULT    = 4.0;   // boosted vs lusion's 1.0 since we have no scene behind us
  var SHADE         = 1.25;

  // ── render loop ───────────────────────────────────────────────────────────

  function tick() {
    requestAnimationFrame(tick);

    var slots   = [ping, pong];
    var prevIdx = curr;
    curr        = 1 - curr;

    var readSlot  = slots[prevIdx];
    var writeSlot = slots[curr];

    // ── simulation pass ────────────────────────────────────────────
    gl.bindFramebuffer(gl.FRAMEBUFFER, writeSlot.fbo);
    gl.viewport(0, 0, simW, simH);

    gl.useProgram(simProg);
    bindQuad(simProg);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, readSlot.tex);
    gl.uniform1i(gl.getUniformLocation(simProg, 'u_prev'), 0);
    gl.uniform2f(gl.getUniformLocation(simProg, 'u_texelSize'), 1 / simW, 1 / simH);
    gl.uniform1f(gl.getUniformLocation(simProg, 'u_pushStrength'), PUSH_STRENGTH);
    gl.uniform3f(gl.getUniformLocation(simProg, 'u_dissipations'), DISS_VEL, DISS_W1, DISS_W2);

    // Convert screen coords → sim pixel coords (flip Y: WebGL origin bottom-left)
    var fromX = prevX  / w * simW;
    var fromY = (1 - prevY  / h) * simH;
    var toX   = mouseX / w * simW;
    var toY   = (1 - mouseY / h) * simH;

    gl.uniform4f(gl.getUniformLocation(simProg, 'u_from'), fromX, fromY, RADIUS, WEIGHT);
    gl.uniform4f(gl.getUniformLocation(simProg, 'u_to'),   toX,   toY,   RADIUS, WEIGHT);

    // Velocity direction + magnitude in simulation UV space
    var speed = Math.sqrt(velX * velX + velY * velY);
    var nx = speed > 0 ? velX / speed : 0;
    var ny = speed > 0 ? velY / speed : 0;
    var mag = Math.min(speed / Math.max(w, h) * 8, 0.08);
    gl.uniform2f(gl.getUniformLocation(simProg, 'u_vel'), nx * mag, -ny * mag);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    // ── display pass ───────────────────────────────────────────────
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, w, h);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(dispProg);
    bindQuad(dispProg);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, writeSlot.tex);
    gl.uniform1i(gl.getUniformLocation(dispProg, 'u_paint'), 0);
    gl.uniform1f(gl.getUniformLocation(dispProg, 'u_rgbShift'),      RGB_SHIFT);
    gl.uniform1f(gl.getUniformLocation(dispProg, 'u_colorMultiplier'), COLOR_MULT);
    gl.uniform1f(gl.getUniformLocation(dispProg, 'u_shade'),         SHADE);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  requestAnimationFrame(tick);
}());
