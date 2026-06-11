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
  function heat(x){ var a=Math.min(1,x*1.1); return "rgba(0,115,230,"+(0.05+0.5*a).toFixed(3)+")"; }
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

  // tabellone previsto (scenario piu' probabile) con interattivita su hover
  function renderBracket(s){
    var box=document.getElementById("bracket"); box.innerHTML="";
    var br=s.predicted_bracket; if(!br){ box.appendChild(el("p","muted","Non disponibile per questa data.")); return; }
    var champ=s.predicted_champion;
    var byTeam={}; s.teams.forEach(function(t){ byTeam[t.team]=t; });
    box.appendChild(el("p","muted small","Scenario piu probabile partita per partita (il favorito avanza). Passa il mouse su una squadra per evidenziare il suo percorso e vedere le sue probabilita per fase."));
    var wrap=el("div","bracket-scroll");
    function teamCell(team, isWin){
      var d=el("div","brk-team"+(isWin?" win":""));
      if(team){ d.textContent=name(team); d.setAttribute("data-team", team); }
      else d.textContent="-";
      return d;
    }
    br.forEach(function(rd){
      var col=el("div","brk-col");
      col.appendChild(el("div","brk-head",rd.round));
      rd.matches.forEach(function(mm){
        var b=el("div","brk-match");
        b.appendChild(teamCell(mm.home, mm.winner===mm.home));
        b.appendChild(teamCell(mm.away, mm.winner===mm.away));
        b.appendChild(el("div","brk-p", mm.p?pctShort(mm.p)+" passa "+it(mm.winner):""));
        col.appendChild(b);
      });
      wrap.appendChild(col);
    });
    var cc=el("div","brk-col");
    cc.appendChild(el("div","brk-head","Campione previsto"));
    var champEl=teamCell(champ, true); champEl.className="brk-champ"; champEl.setAttribute("data-team",champ);
    cc.appendChild(champEl);
    wrap.appendChild(cc);
    box.appendChild(wrap);

    // tooltip + highlight percorso
    var tip=document.getElementById("brkTip");
    if(!tip){ tip=el("div"); tip.id="brkTip"; tip.className="brk-tip"; document.body.appendChild(tip); }
    function show(team, x, y){
      var t=byTeam[team]; if(!t) return;
      tip.innerHTML="<div class='tt-h'>"+name(team)+"</div>"+
        "<div class='tt-r'><span>Supera i gironi</span><b>"+pctShort(t.p_advance)+"</b></div>"+
        "<div class='tt-r'><span>Ottavi</span><b>"+pctShort(t.p_r16)+"</b></div>"+
        "<div class='tt-r'><span>Quarti</span><b>"+pctShort(t.p_qf)+"</b></div>"+
        "<div class='tt-r'><span>Semifinale</span><b>"+pctShort(t.p_sf)+"</b></div>"+
        "<div class='tt-r'><span>Finale</span><b>"+pctShort(t.p_final)+"</b></div>"+
        "<div class='tt-r tt-win'><span>Titolo</span><b>"+pct(t.p_champion)+"</b></div>";
      tip.style.display="block";
      var vw=window.innerWidth;
      tip.style.left=Math.min(x+14, vw-180)+"px";
      tip.style.top=(y+window.scrollY+12)+"px";
    }
    box.querySelectorAll("[data-team]").forEach(function(node){
      node.addEventListener("mouseenter", function(e){
        var team=node.getAttribute("data-team");
        box.querySelectorAll("[data-team='"+(window.CSS&&CSS.escape?CSS.escape(team):team)+"']").forEach(function(n){n.classList.add("brk-hl");});
        show(team, e.clientX, e.clientY);
      });
      node.addEventListener("mousemove", function(e){ show(node.getAttribute("data-team"), e.clientX, e.clientY); });
      node.addEventListener("mouseleave", function(){
        box.querySelectorAll(".brk-hl").forEach(function(n){n.classList.remove("brk-hl");});
        tip.style.display="none";
      });
    });
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
      function seg(cls,p,lab,title){
        var txt = p>=0.14 ? lab+" "+pctShort(p) : (p>=0.07 ? pctShort(p) : "");
        return "<span style='flex:"+Math.max(p,0.05)+"' class='"+cls+"' title='"+title+"'>"+txt+"</span>";
      }
      var split=el("div","msplit");
      split.innerHTML=seg("s1",m.p_home,"1","Vittoria "+it(m.home))+
                      seg("sx",m.p_draw,"X","Pareggio")+
                      seg("s2",m.p_away,"2","Vittoria "+it(m.away));
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
    if(played.length){
      var hit=played.filter(function(m){return m.hit;}).length;
      document.getElementById("playedScore").textContent =
        "Segno azzeccato in "+hit+" partite su "+played.length+" giocate finora.";
      played.forEach(function(m){ bp.appendChild(matchCard(m)); });
    } else {
      document.getElementById("playedScore").textContent="";
      bp.appendChild(el("p","muted empty","Il torneo inizia oggi. Appena le partite vengono giocate, qui compaiono il risultato reale e la predizione del modello, con il segno ✓ (indovinato) o ✗. La sezione si aggiorna ogni giorno in automatico."));
    }

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
      svg+='<line x1="'+pad.l+'" y1="'+y+'" x2="'+(W-pad.r)+'" y2="'+y+'" stroke="#e2e8f0"/>';
      svg+='<text x="'+(pad.l-6)+'" y="'+(y+4)+'" fill="#64748b" font-size="11" text-anchor="end">'+Math.round(100*v)+'%</text>'; }
    var step=Math.max(1,Math.ceil(n/6));
    for(var i=0;i<n;i+=step){ var x=X(i); svg+='<text x="'+x+'" y="'+(H-pad.b+16)+'" fill="#64748b" font-size="10" text-anchor="middle">'+dates[i].slice(5)+'</text>'; }
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

    // pannello: le nuove feature fatte girare sui dati passati
    var vf=state.retro.validated_features;
    if(vf && vf.length){
      var pan=el("div","retro-card"); pan.style.gridColumn="1 / -1";
      pan.appendChild(el("div","retro-head","Le nuove feature, misurate sui dati passati"));
      var t=el("table","bt-tbl");
      t.innerHTML="<tr><th style='text-align:left'>Feature</th><th>Su quali partite</th><th>Log loss prima</th><th>Log loss dopo</th></tr>";
      vf.forEach(function(f){
        var tr=el("tr");
        tr.innerHTML="<td style='text-align:left'>"+f.name+"</td><td style='font-size:.78rem'>"+f.scope+"</td>"+
          "<td>"+f.base_logloss.toFixed(3).replace(".",",")+"</td>"+
          "<td><b>"+f.enh_logloss.toFixed(3).replace(".",",")+"</b></td>";
        t.appendChild(tr);
      });
      pan.appendChild(t);
      var coef=state.retro.squad_coef;
      pan.appendChild(el("p","muted small","Altitudine e forza-rosa migliorano la previsione a livello di <strong>singola partita</strong> sui dati storici (numeri sopra). A livello di <strong>campione del torneo</strong>, pero, non aggiungono: la calibrazione del peso forza-rosa sui Mondiali 2018/2022 sceglie "+(coef===0?"zero":coef)+", perche l'Elo cattura gia la forza della rosa. Per questo i pronostici-campione qui sotto restano quelli del modello Elo+Poisson, onesti e immutati."));
      box.appendChild(pan);
    }
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
      var ex=bt.extra||{};
      var ft=document.getElementById("favoriteTable");
      if(ft && ex.by_favorite){
        var t3=el("table","bt-tbl");
        t3.innerHTML="<tr><th>Tipo di partita</th><th>Partite</th><th>Accuratezza</th><th>Log loss</th></tr>";
        ex.by_favorite.forEach(function(r){
          var tr=el("tr");
          tr.innerHTML="<td style='text-align:left'>"+r.band+"</td><td>"+r.n+"</td><td>"+Math.round(100*r.accuracy)+"%</td><td>"+r.logloss.toFixed(3).replace(".",",")+"</td>";
          t3.appendChild(tr);
        });
        ft.innerHTML=""; ft.appendChild(t3);
      }
      var dn=document.getElementById("drawNote");
      if(dn && ex.draws){
        var d=ex.draws;
        dn.innerHTML="<strong>Pareggi:</strong> il "+Math.round(100*d.share_real)+"% delle partite dei Mondiali finisce in parita. Il modello assegna in media "+Math.round(100*d.avg_p_draw_on_draws)+"% di probabilita al pareggio quando poi succede, ma quasi mai il pareggio e l'esito singolo piu probabile: e una proprieta nota dei modelli di gol (la X resta divisa tra le due squadre). Per questo si valuta su probabilita (log loss, RPS), non solo sul segno secco.";
      }
      var sa=document.getElementById("stageAccuracy");
      if(sa && bt.by_stage){
        var t2=el("table","bt-tbl");
        t2.innerHTML="<tr><th>Fase</th><th>Partite</th><th>Azzeccate</th><th>Accuratezza</th></tr>";
        bt.by_stage.forEach(function(r){
          var tr=el("tr",r.stage==="Tutte"?"pooled":"");
          tr.innerHTML="<td>"+r.stage+"</td><td>"+r.n+"</td><td>"+r.correct+"</td><td>"+Math.round(100*r.accuracy)+"%</td>";
          t2.appendChild(tr);
        });
        sa.innerHTML=""; sa.appendChild(t2);
      }
    }).catch(function(){});
    renderPhase2();
  }

  function renderPhase2(){
    var box=document.getElementById("phase2Table"); if(!box) return;
    getJSON(DATA_BASE+"/phase2.json").then(function(d){
      var t=el("table","bt-tbl");
      t.innerHTML="<tr><th>Modello</th><th>Log loss (pooled)</th><th>RPS</th><th>Log loss (test 18-22)</th><th>RPS (test)</th></tr>";
      d.rows.forEach(function(r){
        var tr=el("tr",r.key==="ensemble"?"pooled":"");
        function f(x){return x.toFixed(4).replace(".",",");}
        tr.innerHTML="<td style='text-align:left'>"+r.model+"</td>"+
          "<td>"+f(r.pooled.logloss)+"</td><td>"+f(r.pooled.rps)+"</td>"+
          "<td>"+f(r.test.logloss)+"</td><td>"+f(r.test.rps)+"</td>";
        t.appendChild(tr);
      });
      box.innerHTML=""; box.appendChild(t);
    }).catch(function(){ box.innerHTML="<p class='muted small'>Confronto fase 2 non disponibile.</p>"; });
  }

  if(document.readyState!=="loading") init();
  else document.addEventListener("DOMContentLoaded", init);
})();
