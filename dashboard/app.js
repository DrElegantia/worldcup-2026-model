/* Dashboard Mondiale 2026 - modello predittivo. Vanilla JS, nessuna dipendenza. */
(function () {
  "use strict";
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
    "Iraq":"🇮🇶","Jordan":"🇯🇴","Haiti":"🇭🇹","Curaçao":"🇨🇼","Pareggio":"⚖️"
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
    "New Zealand":"Nuova Zelanda","Iraq":"Iraq","Jordan":"Giordania","Haiti":"Haiti",
    "Curaçao":"Curaçao","Pareggio":"Pareggio"
  };
  function it(t){ return IT[t]||t; }
  function name(t){ return (FLAG[t]?FLAG[t]+" ":"") + it(t); }
  function pct(x){ return (100*x).toFixed(1).replace(".",",")+"%"; }
  function pctShort(x){ return Math.round(100*x)+"%"; }
  function el(tag, cls, html){ var e=document.createElement(tag); if(cls)e.className=cls; if(html!=null)e.innerHTML=html; return e; }
  function heat(x){ var a=Math.min(1,x*1.15); return "rgba("+Math.round(20+200*(1-a))+","+Math.round(90+120*a)+",80,"+(0.12+0.78*a)+")"; }
  function getJSON(u){ return fetch(u,{cache:"no-store"}).then(function(r){ if(!r.ok) throw new Error(u+" "+r.status); return r.json(); }); }

  var state = { index:null, snap:null, history:null, retro:null };

  function init(){
    getJSON(DATA_BASE+"/index.json").then(function(idx){
      state.index=idx;
      var sel=document.getElementById("dateSelect");
      idx.dates.slice().reverse().forEach(function(d){
        var o=el("option"); o.value=d; o.textContent=(d===idx.latest? d+"  (oggi)": d); sel.appendChild(o);
      });
      sel.value=idx.latest;
      sel.addEventListener("change",function(){ loadSnap(sel.value); });
      document.getElementById("meta").textContent =
        "Modello "+idx.model_version+" · aggiornato "+(idx.updated_utc||"").slice(0,10)+" · "+idx.dates.length+" simulazioni archiviate";
      return loadSnap(idx.latest);
    }).then(function(){ return getJSON(DATA_BASE+"/history.json"); })
      .then(function(h){ state.history=h; })
      .then(function(){ return getJSON(DATA_BASE+"/retro.json").catch(function(){return null;}); })
      .then(function(r){ state.retro=r; renderRetro(); renderBacktest(); })
      .catch(function(e){ document.getElementById("app").innerHTML="<p class='err'>Errore caricamento dati: "+e.message+"</p>"; });

    document.querySelectorAll(".tab").forEach(function(b){
      b.addEventListener("click",function(){
        document.querySelectorAll(".tab").forEach(function(x){x.classList.remove("on");});
        document.querySelectorAll(".view").forEach(function(x){x.classList.remove("on");});
        b.classList.add("on"); document.getElementById(b.dataset.view).classList.add("on");
        if(b.dataset.view==="vTrend") drawTrend();
      });
    });
    document.getElementById("trendMetric").addEventListener("change", drawTrend);
  }

  function loadSnap(date){
    return getJSON(DATA_BASE+"/"+date+".json").then(function(s){ state.snap=s; renderSim(s); });
  }

  function renderSim(s){
    renderContenders(s); renderPredictedGroups(s); renderBracket(s);
    renderKnockout(s); renderMatches(s);
    var played=s.group_matches_played;
    document.getElementById("simStatus").textContent =
      (played===0 ? "Pre-torneo: nessuna partita giocata, simulazione sui rating attuali."
                  : played+" partite dei gironi giocate, simulazione aggiornata sui risultati reali.")
      + " " + (s.n_sims||0).toLocaleString("it-IT") + " simulazioni Monte Carlo.";
  }

  function renderContenders(s){
    var box=document.getElementById("contenders"); box.innerHTML="";
    var top=s.teams.slice(0,16); var max=top[0].p_champion||0.0001;
    top.forEach(function(t){
      var row=el("div","bar-row");
      row.appendChild(el("div","bar-lab",name(t.team)));
      var track=el("div","bar-track"); var fill=el("div","bar-fill"); fill.style.width=(100*t.p_champion/max)+"%";
      track.appendChild(fill); row.appendChild(track);
      row.appendChild(el("div","bar-val",pct(t.p_champion)));
      box.appendChild(row);
    });
  }

  // classifica prevista per girone (ordinata 1-4 con punti attesi)
  function renderPredictedGroups(s){
    var box=document.getElementById("groups"); box.innerHTML="";
    var std=s.predicted_standings||{};
    var advBy={}; s.teams.forEach(function(t){ advBy[t.team]=t; });
    Object.keys(s.groups).forEach(function(g){
      var card=el("div","gcard");
      card.appendChild(el("div","ghead","Girone "+g));
      var tbl=el("table","gtable");
      tbl.innerHTML="<tr><th>#</th><th></th><th title='Punti attesi nei gironi'>Pt att.</th><th title='Probabilita di superare il girone'>Passa</th></tr>";
      (std[g]||[]).forEach(function(r){
        var t=advBy[r.team]||{}; var adv=t.p_advance||0;
        var cls=r.pos<=2?"qual":(adv>=0.5?"qual":"out");
        var tr=el("tr",cls);
        tr.innerHTML="<td class='pos'>"+r.pos+"</td>"+
          "<td class='tn'>"+name(r.team)+"</td>"+
          "<td>"+(r.exp_points!=null?r.exp_points.toFixed(1).replace(".",","):"-")+"</td>"+
          "<td class='q' style='background:"+heat(adv)+"'>"+pctShort(adv)+"</td>";
        tbl.appendChild(tr);
      });
      card.appendChild(tbl);
      box.appendChild(card);
    });
  }

  // tabellone previsto (scenario piu' probabile)
  function renderBracket(s){
    var box=document.getElementById("bracket"); box.innerHTML="";
    var br=s.predicted_bracket; if(!br){ box.appendChild(el("p","muted","Non disponibile per questa data.")); return; }
    var champ=s.predicted_champion;
    box.appendChild(el("p","muted small","Scenario piu' probabile partita per partita (il favorito avanza). La squadra in grassetto e il favorito, con la sua probabilita di passare il turno."));
    var wrap=el("div","bracket-scroll");
    br.forEach(function(rd){
      var col=el("div","brk-col");
      col.appendChild(el("div","brk-head",rd.round));
      rd.matches.forEach(function(mm){
        var b=el("div","brk-match");
        var w=mm.winner;
        b.innerHTML=
          "<div class='brk-team "+(w===mm.home?"win":"")+"'>"+(mm.home?name(mm.home):"-")+"</div>"+
          "<div class='brk-team "+(w===mm.away?"win":"")+"'>"+(mm.away?name(mm.away):"-")+"</div>"+
          "<div class='brk-p'>"+(mm.p?pctShort(mm.p)+" passa "+it(w):"")+"</div>";
        col.appendChild(b);
      });
      wrap.appendChild(col);
    });
    // colonna campione
    var cc=el("div","brk-col");
    cc.appendChild(el("div","brk-head","Campione previsto"));
    cc.appendChild(el("div","brk-champ",name(champ)));
    wrap.appendChild(cc);
    box.appendChild(wrap);
  }

  function renderKnockout(s){
    var box=document.getElementById("knockout"); box.innerHTML="";
    var cols=[["p_advance","Sedicesimi"],["p_r16","Ottavi"],["p_qf","Quarti"],["p_sf","Semi"],["p_final","Finale"],["p_champion","Titolo"]];
    var tbl=el("table","ktable");
    var head="<tr><th>Squadra</th>"; cols.forEach(function(c){head+="<th>"+c[1]+"</th>";}); head+="</tr>";
    tbl.innerHTML=head;
    s.teams.slice(0,24).forEach(function(t){
      var tr=el("tr"); var h="<td class='tn'>"+name(t.team)+"</td>";
      cols.forEach(function(c){ var v=t[c[0]]||0; h+="<td style='background:"+heat(v)+"'>"+pctShort(v)+"</td>"; });
      tr.innerHTML=h; tbl.appendChild(tr);
    });
    box.appendChild(tbl);
  }

  // legenda + card partita chiare; predetto vs reale per le giocate
  function matchCard(m){
    var row=el("div","match");
    var hdr=m.date.slice(5)+(m.group?" · Girone "+m.group:" · Eliminazione")+" · "+m.city;
    row.appendChild(el("div","mdate",hdr));
    var line=el("div","mteams");
    if(m.played){
      line.innerHTML="<span class='mh'>"+name(m.home)+"</span>"+
        "<span class='mscore'>"+m.home_score+" - "+m.away_score+"</span>"+
        "<span class='ma'>"+name(m.away)+"</span>";
      row.appendChild(line);
      var ok=m.hit;
      var note=el("div","mpred");
      note.innerHTML="<span class='"+(ok?"hit":"miss")+"'>"+(ok?"✓ previsto":"✗ non previsto")+"</span> "+
        "<span class='muted'>modello: "+m.pred_score+" ("+it(m.pred_winner)+")</span>";
      row.appendChild(note);
    } else {
      line.innerHTML="<span class='mh'>"+name(m.home)+"</span>"+
        "<span class='mvs'>vs</span>"+
        "<span class='ma'>"+name(m.away)+"</span>";
      row.appendChild(line);
      row.appendChild(el("div","mpred","<span class='muted'>Risultato piu probabile:</span> <strong>"+m.pred_score+"</strong> ("+it(m.pred_winner)+")"));
      var split=el("div","msplit");
      split.innerHTML="<span style='flex:"+Math.max(m.p_home,0.02)+"' class='s1' title='Vittoria "+it(m.home)+"'>1 "+pctShort(m.p_home)+"</span>"+
                      "<span style='flex:"+Math.max(m.p_draw,0.02)+"' class='sx' title='Pareggio'>X "+pctShort(m.p_draw)+"</span>"+
                      "<span style='flex:"+Math.max(m.p_away,0.02)+"' class='s2' title='Vittoria "+it(m.away)+"'>2 "+pctShort(m.p_away)+"</span>";
      row.appendChild(split);
    }
    return row;
  }

  function renderMatches(s){
    var ms=s.matches||[];
    var n48=ms.filter(function(m){return m.within_48h && !m.played;});
    var played=ms.filter(function(m){return m.played;}).slice(-12).reverse();
    var next=ms.filter(function(m){return !m.played && !m.within_48h;}).slice(0,12);

    var b48=document.getElementById("next48"); b48.innerHTML="";
    if(n48.length){ n48.forEach(function(m){ b48.appendChild(matchCard(m)); }); }
    else { b48.appendChild(el("p","muted","Nessuna partita nelle prossime 48 ore.")); }

    var bp=document.getElementById("playedBox"); bp.innerHTML="";
    var sec=document.getElementById("playedSection");
    if(played.length){ sec.style.display=""; played.forEach(function(m){ bp.appendChild(matchCard(m)); }); }
    else { sec.style.display="none"; }

    var bn=document.getElementById("nextBox"); bn.innerHTML="";
    if(next.length){ next.forEach(function(m){ bn.appendChild(matchCard(m)); }); }
    else { bn.appendChild(el("p","muted","-")); }
  }

  // ---- andamento day-by-day ----
  var PALETTE=["#e63946","#457b9d","#2a9d8f","#e9c46a","#f4a261","#9b5de5","#00bbf9","#fb6f92"];
  function drawTrend(){
    if(!state.history) return;
    var metric=document.getElementById("trendMetric").value;
    var h=state.history, dates=h.dates;
    var teams=h.teams.slice().sort(function(a,b){ return (b[metric][b[metric].length-1]||0)-(a[metric][a[metric].length-1]||0); }).slice(0,8);
    var cw=document.getElementById("trendChart").clientWidth||document.getElementById("vTrend").clientWidth||760;
    var W=Math.max(360,Math.min(900,cw-10)), H=420, pad={l:44,r:150,t:20,b:46};
    var maxV=0.05; teams.forEach(function(t){ t[metric].forEach(function(v){ if(v>maxV)maxV=v; }); });
    var n=dates.length;
    function X(i){ return pad.l+(n<=1?0:i*(W-pad.l-pad.r)/(n-1)); }
    function Y(v){ return H-pad.b-v/maxV*(H-pad.t-pad.b); }
    var svg='<svg viewBox="0 0 '+W+' '+H+'" width="100%" preserveAspectRatio="xMidYMid meet">';
    for(var gy=0;gy<=4;gy++){ var v=maxV*gy/4,y=Y(v);
      svg+='<line x1="'+pad.l+'" y1="'+y+'" x2="'+(W-pad.r)+'" y2="'+y+'" stroke="#26303a"/>';
      svg+='<text x="'+(pad.l-6)+'" y="'+(y+4)+'" fill="#7d8b99" font-size="11" text-anchor="end">'+Math.round(100*v)+'%</text>'; }
    var step=Math.max(1,Math.ceil(n/6));
    for(var i=0;i<n;i+=step){ var x=X(i); svg+='<text x="'+x+'" y="'+(H-pad.b+16)+'" fill="#7d8b99" font-size="10" text-anchor="middle">'+dates[i].slice(5)+'</text>'; }
    teams.forEach(function(t,k){
      var col=PALETTE[k%PALETTE.length],d="";
      t[metric].forEach(function(v,i){ d+=(i?"L":"M")+X(i).toFixed(1)+" "+Y(v).toFixed(1)+" "; });
      svg+='<path d="'+d+'" fill="none" stroke="'+col+'" stroke-width="2.5"/>';
      var lv=t[metric][t[metric].length-1];
      svg+='<circle cx="'+X(n-1)+'" cy="'+Y(lv)+'" r="3" fill="'+col+'"/>';
      svg+='<text x="'+(W-pad.r+8)+'" y="'+(Y(lv)+4)+'" fill="'+col+'" font-size="11">'+name(t.team)+' '+pctShort(lv)+'</text>';
    });
    svg+='</svg>';
    document.getElementById("trendChart").innerHTML=svg;
  }

  // ---- verifica storica ----
  function renderRetro(){
    var box=document.getElementById("retroBox"); if(!box) return; box.innerHTML="";
    if(!state.retro){ box.appendChild(el("p","muted","Verifica storica non disponibile.")); return; }
    box.appendChild(el("p","muted","Per ogni Mondiale passato, le probabilita che il modello assegnava PRIMA del torneo (solo dati anteriori), confrontate con l'esito reale. Verifica leale: il favorito del modello non sempre vince, ma il campione reale era quasi sempre fra i primissimi."));
    state.retro.tournaments.forEach(function(tn){
      var card=el("div","retro-card");
      var a=tn.actual;
      card.appendChild(el("div","retro-head","Mondiale "+tn.year));
      var win=el("div","retro-actual");
      win.innerHTML="Campione reale: <strong>"+name(a.champion)+"</strong> · finalista "+name(a.finalist)+
        "<br><span class='hl'>Il modello lo dava al posto #"+tn.champion_pred_rank+" ("+pct(tn.champion_pred_prob)+" titolo)</span>";
      card.appendChild(win);
      var tbl=el("table","retro-tbl");
      tbl.innerHTML="<tr><th>#</th><th>Favorita pre-torneo</th><th>Titolo</th></tr>";
      tn.top.forEach(function(r,i){
        var isChamp=r.team===a.champion;
        var tr=el("tr",isChamp?"champ":"");
        tr.innerHTML="<td>"+(i+1)+"</td><td class='tn'>"+name(r.team)+(isChamp?" 🏆":"")+"</td><td>"+pct(r.p_champion)+"</td>";
        tbl.appendChild(tr);
      });
      card.appendChild(tbl);
      box.appendChild(card);
    });
  }

  // tabella backtest nella nota metodologica
  function renderBacktest(){
    var box=document.getElementById("backtestTable"); if(!box) return;
    getJSON(DATA_BASE+"/backtest.json").then(function(bt){
      var rows=bt.summary||[];
      var tbl=el("table","bt-tbl");
      tbl.innerHTML="<tr><th>Mondiale</th><th>Partite</th><th>Log loss</th><th>RPS</th><th>Brier</th><th>Accuratezza</th></tr>";
      rows.forEach(function(r){
        var tr=el("tr",r.year==="POOLED"?"pooled":"");
        tr.innerHTML="<td>"+r.year+"</td><td>"+r.n_matches+"</td>"+
          "<td>"+(r.logloss_model).toFixed(3).replace(".",",")+"</td>"+
          "<td>"+(r.rps_model).toFixed(3).replace(".",",")+"</td>"+
          "<td>"+(r.brier_model).toFixed(3).replace(".",",")+"</td>"+
          "<td>"+Math.round(100*r.acc_model)+"%</td>";
        tbl.appendChild(tr);
      });
      box.innerHTML=""; box.appendChild(tbl);
    }).catch(function(){});
  }

  if(document.readyState!=="loading") init();
  else document.addEventListener("DOMContentLoaded", init);
})();
