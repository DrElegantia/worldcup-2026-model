/* Dashboard Mondiale 2026 - modello predittivo. Vanilla JS, nessuna dipendenza. */
(function () {
  "use strict";

  // base dati: relativo (Pages) oppure assoluto se servito da altrove
  var DATA_BASE = (window.WC_DATA_BASE || "./data").replace(/\/$/, "");

  var FLAG = {
    "Spain":"🇪🇸","Argentina":"🇦🇷","France":"🇫🇷","England":"🏴󠁧󠁢󠁥󠁮󠁧󠁿","Brazil":"🇧🇷",
    "Colombia":"🇨🇴","Portugal":"🇵🇹","Germany":"🇩🇪","Netherlands":"🇳🇱","Ecuador":"🇪🇨",
    "Japan":"🇯🇵","Morocco":"🇲🇦","Norway":"🇳🇴","Croatia":"🇭🇷","Turkey":"🇹🇷",
    "Uruguay":"🇺🇾","Switzerland":"🇨🇭","Mexico":"🇲🇽","Belgium":"🇧🇪","United States":"🇺🇸",
    "Senegal":"🇸🇳","Iran":"🇮🇷","Egypt":"🇪🇬","South Korea":"🇰🇷","Canada":"🇨🇦",
    "Australia":"🇦🇺","Ivory Coast":"🇨🇮","Austria":"🇦🇹","Sweden":"🇸🇪","Paraguay":"🇵🇾",
    "Scotland":"🏴󠁧󠁢󠁳󠁣󠁴󠁿","Algeria":"🇩🇿","Qatar":"🇶🇦","Panama":"🇵🇦","Tunisia":"🇹🇳",
    "Saudi Arabia":"🇸🇦","Ghana":"🇬🇭","Cape Verde":"🇨🇻","DR Congo":"🇨🇩","Uzbekistan":"🇺🇿",
    "Czech Republic":"🇨🇿","South Africa":"🇿🇦","Bosnia and Herzegovina":"🇧🇦","New Zealand":"🇳🇿",
    "Iraq":"🇮🇶","Jordan":"🇯🇴","Haiti":"🇭🇹","Curaçao":"🇨🇼"
  };
  var IT = {
    "Spain":"Spagna","Argentina":"Argentina","France":"Francia","England":"Inghilterra",
    "Brazil":"Brasile","Colombia":"Colombia","Portugal":"Portogallo","Germany":"Germania",
    "Netherlands":"Olanda","Ecuador":"Ecuador","Japan":"Giappone","Morocco":"Marocco",
    "Norway":"Norvegia","Croatia":"Croazia","Turkey":"Turchia","Uruguay":"Uruguay",
    "Switzerland":"Svizzera","Mexico":"Messico","Belgium":"Belgio","United States":"USA",
    "Senegal":"Senegal","Iran":"Iran","Egypt":"Egitto","South Korea":"Corea del Sud",
    "Canada":"Canada","Australia":"Australia","Ivory Coast":"Costa d'Avorio","Austria":"Austria",
    "Sweden":"Svezia","Paraguay":"Paraguay","Scotland":"Scozia","Algeria":"Algeria",
    "Qatar":"Qatar","Panama":"Panama","Tunisia":"Tunisia","Saudi Arabia":"Arabia Saudita",
    "Ghana":"Ghana","Cape Verde":"Capo Verde","DR Congo":"RD Congo","Uzbekistan":"Uzbekistan",
    "Czech Republic":"Cechia","South Africa":"Sudafrica","Bosnia and Herzegovina":"Bosnia",
    "New Zealand":"Nuova Zelanda","Iraq":"Iraq","Jordan":"Giordania","Haiti":"Haiti","Curaçao":"Curaçao"
  };
  function name(t){ return (FLAG[t]?FLAG[t]+" ":"") + (IT[t]||t); }
  function pct(x){ return (100*x).toFixed(1).replace(".",",")+"%"; }
  function pctShort(x){ return Math.round(100*x)+"%"; }

  function el(tag, cls, html){ var e=document.createElement(tag); if(cls)e.className=cls; if(html!=null)e.innerHTML=html; return e; }
  function heat(x){ // 0..1 -> colore (verde piu' alto)
    var a = Math.min(1, x*1.15);
    return "rgba("+Math.round(20+200*(1-a))+","+Math.round(90+120*a)+","+Math.round(80)+","+(0.12+0.78*a)+")";
  }
  function getJSON(url){ return fetch(url, {cache:"no-store"}).then(function(r){ if(!r.ok) throw new Error(url+" "+r.status); return r.json(); }); }

  var state = { index:null, snap:null, history:null };

  function init(){
    getJSON(DATA_BASE+"/index.json").then(function(idx){
      state.index = idx;
      var sel = document.getElementById("dateSelect");
      idx.dates.slice().reverse().forEach(function(d){
        var o=el("option"); o.value=d; o.textContent=(d===idx.latest? d+"  (oggi)": d); sel.appendChild(o);
      });
      sel.value = idx.latest;
      sel.addEventListener("change", function(){ loadSnap(sel.value); });
      document.getElementById("meta").textContent =
        "Modello "+idx.model_version+" · aggiornato "+(idx.updated_utc||"").slice(0,10)+" · "+idx.dates.length+" simulazioni archiviate";
      return loadSnap(idx.latest);
    }).then(function(){
      return getJSON(DATA_BASE+"/history.json");
    }).then(function(h){ state.history=h; drawTrend(); })
      .catch(function(e){ document.getElementById("app").innerHTML =
        "<p class='err'>Impossibile caricare i dati: "+e.message+"</p>"; });

    document.querySelectorAll(".tab").forEach(function(b){
      b.addEventListener("click", function(){
        document.querySelectorAll(".tab").forEach(function(x){x.classList.remove("on");});
        document.querySelectorAll(".view").forEach(function(x){x.classList.remove("on");});
        b.classList.add("on");
        document.getElementById(b.dataset.view).classList.add("on");
        if(b.dataset.view==="vTrend") drawTrend();
      });
    });
    document.getElementById("trendMetric").addEventListener("change", drawTrend);
  }

  function loadSnap(date){
    return getJSON(DATA_BASE+"/"+date+".json").then(function(s){
      state.snap=s; renderSim(s);
    });
  }

  function renderSim(s){
    renderContenders(s);
    renderGroups(s);
    renderKnockout(s);
    renderUpcoming(s);
    var played = s.group_matches_played;
    document.getElementById("simStatus").textContent =
      (played===0 ? "Pre-torneo: nessuna partita giocata, simulazione sui rating attuali."
                  : played+" partite dei gironi giocate, simulazione aggiornata sui risultati reali.")
      + " " + s.n_sims.toLocaleString("it-IT") + " iterazioni Monte Carlo.";
  }

  function renderContenders(s){
    var box=document.getElementById("contenders"); box.innerHTML="";
    var top=s.teams.slice(0,16);
    var max=top[0].p_champion||0.0001;
    top.forEach(function(t){
      var row=el("div","bar-row");
      row.appendChild(el("div","bar-lab", name(t.team)));
      var track=el("div","bar-track");
      var fill=el("div","bar-fill"); fill.style.width=(100*t.p_champion/max)+"%";
      track.appendChild(fill);
      row.appendChild(track);
      row.appendChild(el("div","bar-val", pct(t.p_champion)));
      box.appendChild(row);
    });
  }

  function renderGroups(s){
    var box=document.getElementById("groups"); box.innerHTML="";
    Object.keys(s.groups).forEach(function(g){
      var card=el("div","gcard");
      card.appendChild(el("div","ghead","Girone "+g));
      var teams=s.teams.filter(function(t){return t.group===g;})
                       .sort(function(a,b){return b.p_advance-a.p_advance;});
      var tbl=el("table","gtable");
      tbl.innerHTML="<tr><th></th><th>1°</th><th>2°</th><th>Qual.</th></tr>";
      teams.forEach(function(t){
        var tr=el("tr");
        tr.innerHTML="<td class='tn'>"+name(t.team)+"</td>"+
          "<td>"+pctShort(t.p_first)+"</td>"+
          "<td>"+pctShort(t.p_second)+"</td>"+
          "<td class='q' style='background:"+heat(t.p_advance)+"'>"+pctShort(t.p_advance)+"</td>";
        tbl.appendChild(tr);
      });
      card.appendChild(tbl);
      box.appendChild(card);
    });
  }

  function renderKnockout(s){
    var box=document.getElementById("knockout"); box.innerHTML="";
    var cols=[["p_advance","Ottavi"],["p_r16","Ottavi→"],["p_qf","Quarti"],
              ["p_sf","Semi"],["p_final","Finale"],["p_champion","Titolo"]];
    var tbl=el("table","ktable");
    var head="<tr><th>Squadra</th>"; cols.forEach(function(c){head+="<th>"+c[1]+"</th>";}); head+="</tr>";
    tbl.innerHTML=head;
    s.teams.slice(0,24).forEach(function(t){
      var tr=el("tr");
      var h="<td class='tn'>"+name(t.team)+"</td>";
      cols.forEach(function(c){ var v=t[c[0]]||0; h+="<td style='background:"+heat(v)+"'>"+pctShort(v)+"</td>"; });
      tr.innerHTML=h; tbl.appendChild(tr);
    });
    box.appendChild(tbl);
  }

  function renderUpcoming(s){
    var box=document.getElementById("upcoming"); box.innerHTML="";
    if(!s.upcoming || !s.upcoming.length){ box.appendChild(el("p","muted","Nessuna partita in programma nei prossimi giorni.")); return; }
    s.upcoming.slice(0,16).forEach(function(m){
      var row=el("div","match");
      row.appendChild(el("div","mdate", m.date.slice(5)+(m.group?" · Gir."+m.group:" · KO")));
      var line=el("div","mteams");
      line.innerHTML="<span class='mh'>"+name(m.home)+"</span><span class='mvs'>"+
        m.xg_home.toFixed(1)+" - "+m.xg_away.toFixed(1)+"</span><span class='ma'>"+name(m.away)+"</span>";
      row.appendChild(line);
      var split=el("div","msplit");
      split.innerHTML="<span style='flex:"+m.p_home+"' class='s1' title='1'>"+pctShort(m.p_home)+"</span>"+
                      "<span style='flex:"+m.p_draw+"' class='sx' title='X'>"+pctShort(m.p_draw)+"</span>"+
                      "<span style='flex:"+m.p_away+"' class='s2' title='2'>"+pctShort(m.p_away)+"</span>";
      row.appendChild(split);
      box.appendChild(row);
    });
  }

  // ---- vista andamento day-by-day (SVG multilinea, nessuna dipendenza) ----
  var PALETTE=["#e63946","#457b9d","#2a9d8f","#e9c46a","#f4a261","#9b5de5","#00bbf9","#fb6f92"];
  function drawTrend(){
    if(!state.history) return;
    var metric=document.getElementById("trendMetric").value;
    var h=state.history, dates=h.dates;
    var teams=h.teams.slice().sort(function(a,b){
      return (b[metric][b[metric].length-1]||0)-(a[metric][a[metric].length-1]||0);
    }).slice(0,8);
    var cw=document.getElementById("trendChart").clientWidth || document.getElementById("vTrend").clientWidth || 760;
    var W=Math.max(360, Math.min(900, cw-10)), H=420;
    var pad={l:44,r:140,t:20,b:46};
    var maxV=0; teams.forEach(function(t){ t[metric].forEach(function(v){ if(v>maxV)maxV=v; }); });
    maxV=Math.max(maxV,0.05);
    var n=dates.length;
    function X(i){ return pad.l + (n<=1?0:i*(W-pad.l-pad.r)/(n-1)); }
    function Y(v){ return H-pad.b - v/maxV*(H-pad.t-pad.b); }
    var svg='<svg viewBox="0 0 '+W+' '+H+'" width="100%" preserveAspectRatio="xMidYMid meet">';
    // griglia Y
    for(var gy=0; gy<=4; gy++){ var v=maxV*gy/4, y=Y(v);
      svg+='<line x1="'+pad.l+'" y1="'+y+'" x2="'+(W-pad.r)+'" y2="'+y+'" stroke="#26303a"/>';
      svg+='<text x="'+(pad.l-6)+'" y="'+(y+4)+'" fill="#7d8b99" font-size="11" text-anchor="end">'+Math.round(100*v)+'%</text>';
    }
    // assi X (mostra alcune date)
    var step=Math.ceil(n/6);
    for(var i=0;i<n;i+=step){ var x=X(i);
      svg+='<text x="'+x+'" y="'+(H-pad.b+16)+'" fill="#7d8b99" font-size="10" text-anchor="middle">'+dates[i].slice(5)+'</text>';
    }
    teams.forEach(function(t,k){
      var col=PALETTE[k%PALETTE.length], d="";
      t[metric].forEach(function(v,i){ d+=(i?"L":"M")+X(i).toFixed(1)+" "+Y(v).toFixed(1)+" "; });
      svg+='<path d="'+d+'" fill="none" stroke="'+col+'" stroke-width="2.5"/>';
      var lastv=t[metric][t[metric].length-1];
      svg+='<circle cx="'+X(n-1)+'" cy="'+Y(lastv)+'" r="3" fill="'+col+'"/>';
      svg+='<text x="'+(W-pad.r+8)+'" y="'+(Y(lastv)+4)+'" fill="'+col+'" font-size="11">'+name(t.team)+' '+pctShort(lastv)+'</text>';
    });
    svg+='</svg>';
    document.getElementById("trendChart").innerHTML=svg;
  }

  if(document.readyState!=="loading") init();
  else document.addEventListener("DOMContentLoaded", init);
})();
