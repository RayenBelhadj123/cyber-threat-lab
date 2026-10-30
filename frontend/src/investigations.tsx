import {useEffect, useMemo, useState, type FormEvent} from 'react';
import {Activity, AlertTriangle, ChevronLeft, ChevronRight, Database, Search, ShieldCheck} from 'lucide-react';
import {Background, Controls, MiniMap, ReactFlow, type Edge, type Node, Position} from '@xyflow/react';

type Dataset='expanded'|'CICIDS2017'|'CICDDoS2019';
type Flow={dataset:string;event_id:string;timestamp:string;source_ip:string;source_port:number;destination_ip:string;destination_port:number;protocol:number;protocol_name:string;duration:number;bytes_total:number;packets_total:number;rate:number;label:string;malicious:boolean;source_file:string;capture_group:string;risk_score:number|null};
type FlowPage={items:Flow[];total:number;page:number;page_size:number;data_mode:string;source:string;indexed_record_count?:number;risk_score_available:boolean};
type FlowStats={dataset:Dataset;total_flows:number;suspicious_flows:number;protocols:Record<string,number>;labels:Record<string,number>;top_sources:{source_ip:string;flows:number}[];top_destination_ports:{port:number;flows:number}[];hourly:{hour:string;flows:number}[];time_range_utc:string[]|null;mean_packets:number|null;mean_duration_microseconds:number|null;data_mode:string;source?:string};
type GraphEdge={dataset:string;source_ip:string;destination_ip:string;destination_port:number;protocol:number;protocol_name:string;flow_count:number;malicious_count:number;bytes_total:number;first_seen:string;last_seen:string};
type InvestigationFilter={source_ip?:string;destination_ip?:string;destination_port?:number};
type NodeData={label:string;filters:InvestigationFilter};

const API=(import.meta.env.VITE_API_URL||'http://127.0.0.1:8000/api').replace(/\/$/,'');
const formatNumber=(value?:number|null)=>typeof value==='number'?new Intl.NumberFormat('en-US').format(value):'—';
const safeError=(value:unknown)=>value instanceof Error?value.message:String(value);

function useJson<T>(path:string,refresh:number,enabled=true){
  const [data,setData]=useState<T|null>(null),[error,setError]=useState(''),[loading,setLoading]=useState(enabled);
  useEffect(()=>{
    if(!enabled){setLoading(false);setError('');return;}
    const controller=new AbortController();setLoading(true);setError('');
    fetch(`${API}${path}`,{signal:controller.signal}).then(async response=>{
      if(!response.ok)throw new Error(`${response.status}: ${(await response.text()).slice(0,220)}`);
      return response.json() as Promise<T>;
    }).then(setData).catch(error=>{if((error as Error).name!=='AbortError')setError(safeError(error));})
      .finally(()=>{if(!controller.signal.aborted)setLoading(false);});
    return()=>controller.abort();
  },[path,refresh,enabled]);
  return {data,error,loading};
}

function queryString(dataset:Dataset,filters:InvestigationFilter){
  const params=new URLSearchParams({dataset,page:'1',page_size:'10',sort:'timestamp',order:'desc'});
  Object.entries(filters).forEach(([key,value])=>{if(value!==undefined&&value!=='')params.set(key,String(value));});
  return `/flows?${params.toString()}`;
}

function FlowRows({items,onInspect,compact=false}:{items:Flow[];onInspect?:(flow:Flow)=>void;compact?:boolean}){
  if(!items.length)return <div className="flow-empty">No records match this query in the selected benchmark scope.</div>;
  return <div className="flow-table-wrap"><table className="flow-table"><thead><tr><th>UTC TIME</th><th>DATASET / SOURCE</th><th>SOURCE IP:PORT</th><th>DESTINATION IP:PORT</th><th>PROTOCOL</th><th>REFERENCE LABEL</th><th>BYTES</th>{!compact&&<th>INSPECT</th>}</tr></thead><tbody>{items.map(flow=><tr key={`${flow.dataset}:${flow.event_id}`}>
    <td>{flow.timestamp.replace('T',' ').replace('+00:00','Z')}</td>
    <td><b>{flow.dataset}</b><small>{flow.source_file}</small></td>
    <td><code>{flow.source_ip}:{flow.source_port}</code></td>
    <td><code>{flow.destination_ip}:{flow.destination_port}</code></td>
    <td>{flow.protocol_name}</td>
    <td><span className={`flow-label ${flow.malicious?'suspicious':''}`}>{flow.label}</span></td>
    <td>{formatNumber(flow.bytes_total)}</td>
    {!compact&&<td><button className="flow-inspect" onClick={()=>onInspect?.(flow)} aria-label="Inspect flow">Inspect</button></td>}
  </tr>)}</tbody></table></div>;
}

function EvidencePanel({flow,onClose}:{flow:Flow|null;onClose:()=>void}){
  if(!flow)return null;
  return <section className="panel flow-evidence-panel"><div className="panel-head"><div><h3>Flow evidence</h3><span className="tiny-label">SOURCE RECORD · HISTORICAL</span></div><button className="text-button" onClick={onClose} aria-label="Close flow evidence">Close</button></div>
    <div className="evidence-list">
      <Evidence label="Record ID" value={flow.event_id}/><Evidence label="Observed time (UTC)" value={flow.timestamp}/>
      <Evidence label="Flow identity" value={`${flow.source_ip}:${flow.source_port} → ${flow.destination_ip}:${flow.destination_port} · ${flow.protocol_name}`}/>
      <Evidence label="Reference label" value={flow.label}/><Evidence label="Provenance" value={`${flow.dataset} · ${flow.source_file} · ${flow.capture_group}`}/>
      <Evidence label="Raw-flow risk score" value="Unavailable; no per-flow score is stored"/>
    </div>
    <div className="callout"><AlertTriangle size={15}/><span>Full-benchmark per-flow model scores are not available. The saved experiment evaluates aggregated host/minute feature windows; legacy Thursday findings are not joined to these raw records.</span></div>
  </section>;
}

function Evidence({label,value}:{label:string;value:string}){return <div className="evidence-row"><span>{label}</span><b>{value}</b></div>;}

export function FlowInvestigation({refresh,featureWindows}:{refresh:number;featureWindows:number}){
  const [dataset,setDataset]=useState<Dataset>('expanded');
  const [draft,setDraft]=useState({q:'',source_ip:'',destination_ip:'',source_port:'',destination_port:'',protocol:'',label:'',start:'',end:''});
  const [filters,setFilters]=useState<Record<string,string>>({});
  const [page,setPage]=useState(1);
  const [sort,setSort]=useState('timestamp');
  const [order,setOrder]=useState<'asc'|'desc'>('desc');
  const [selected,setSelected]=useState<Flow|null>(null);
  const params=useMemo(()=>{const query=new URLSearchParams({dataset,page:String(page),page_size:'50',sort,order});Object.entries(filters).forEach(([key,value])=>{if(value)query.set(key,value);});return `/flows?${query.toString()}`;},[dataset,page,sort,order,filters]);
  const rows=useJson<FlowPage>(params,refresh);
  const stats=useJson<FlowStats>(`/flows/stats?dataset=${dataset}`,refresh);
  const pages=Math.max(1,Math.ceil((rows.data?.total||0)/50));
  const set=(key:keyof typeof draft,value:string)=>setDraft(current=>({...current,[key]:value}));
  const apply=(event:FormEvent)=>{event.preventDefault();const next:Record<string,string>={};Object.entries(draft).forEach(([key,value])=>{if(value)next[key]=value;});
    for(const key of ['start','end'])if(next[key])next[key]=new Date(next[key]).toISOString();
    setPage(1);setSelected(null);setFilters(next);
  };
  return <>
    <section className="panel flow-investigation">
      <div className="panel-head"><div><h2>Full-benchmark flow investigation</h2><span className="tiny-label">SERVER-SIDE SEARCH · RAW FLOWS · HISTORICAL</span></div><span className="flow-provenance"><Database size={14}/> verified local store</span></div>
      <p className="flow-intro">Search normalized raw records from the complete expanded benchmark. Results are paginated on the server; the browser receives no more than 50 rows per page. Feature windows and model metrics remain separate evidence layers.</p>
      <div className="kpi-grid compact flow-kpis">
        <div className="kpi"><div className="kpi-top"><span>Accessible raw flows</span><Activity size={15}/></div><div className="kpi-value">{stats.loading?'…':formatNumber(stats.data?.total_flows)}</div><div className="kpi-helper">{dataset==='expanded'?'CICIDS2017 + CIC-DDoS2019':'Selected dataset only'}</div></div>
        <div className="kpi"><div className="kpi-top"><span>Reference-labeled suspicious</span><AlertTriangle size={15}/></div><div className="kpi-value">{stats.loading?'…':formatNumber(stats.data?.suspicious_flows)}</div><div className="kpi-helper">Dataset labels, not current alerts</div></div>
        <div className="kpi"><div className="kpi-top"><span>Feature windows (separate)</span><ShieldCheck size={15}/></div><div className="kpi-value">{formatNumber(featureWindows)}</div><div className="kpi-helper">Model input rows, not raw flows</div></div>
      </div>
      <form className="flow-filters" onSubmit={apply}>
        <label>Dataset scope<select aria-label="Dataset scope" value={dataset} onChange={event=>{setDataset(event.target.value as Dataset);setPage(1);setSelected(null);}}><option value="expanded">Expanded benchmark (both datasets)</option><option value="CICIDS2017">CICIDS2017 · full week</option><option value="CICDDoS2019">CIC-DDoS2019 · extension</option></select></label>
        <label>Sort results<select aria-label="Sort results" value={sort} onChange={event=>setSort(event.target.value)}><option value="timestamp">Observed time</option><option value="source_ip">Source IP</option><option value="destination_ip">Destination IP</option><option value="source_port">Source port</option><option value="destination_port">Destination port</option><option value="protocol">Protocol</option><option value="label">Attack label</option><option value="bytes">Bytes</option><option value="packets">Packets</option><option value="dataset">Dataset</option></select></label>
        <label>Sort order<select aria-label="Sort order" value={order} onChange={event=>setOrder(event.target.value as 'asc'|'desc')}><option value="desc">Descending</option><option value="asc">Ascending</option></select></label>
        <label>Search IP / label<input aria-label="Search full benchmark flows" value={draft.q} onChange={event=>set('q',event.target.value)} placeholder="Search an IP or label"/></label>
        <label>Source IP<input aria-label="Source IP" value={draft.source_ip} onChange={event=>set('source_ip',event.target.value)} placeholder="Exact IPv4 / IPv6"/></label>
        <label>Destination IP<input aria-label="Destination IP" value={draft.destination_ip} onChange={event=>set('destination_ip',event.target.value)} placeholder="Exact IPv4 / IPv6"/></label>
        <label>Source port<input aria-label="Source port" type="number" min="0" max="65535" value={draft.source_port} onChange={event=>set('source_port',event.target.value)} placeholder="Any"/></label>
        <label>Destination port<input aria-label="Destination port" type="number" min="0" max="65535" value={draft.destination_port} onChange={event=>set('destination_port',event.target.value)} placeholder="Any"/></label>
        <label>Protocol<select aria-label="Protocol" value={draft.protocol} onChange={event=>set('protocol',event.target.value)}><option value="">All protocols</option><option value="6">TCP</option><option value="17">UDP</option><option value="1">ICMP</option></select></label>
        <label>Attack label<input aria-label="Attack label" value={draft.label} onChange={event=>set('label',event.target.value)} placeholder="Recorded source label"/></label>
        <label>From (local time)<input aria-label="Start timestamp" type="datetime-local" value={draft.start} onChange={event=>set('start',event.target.value)}/></label>
        <label>To (local time)<input aria-label="End timestamp" type="datetime-local" value={draft.end} onChange={event=>set('end',event.target.value)}/></label>
        <button className="button-primary" type="submit"><Search size={14}/> Apply filters</button>
        <span className="risk-unavailable">Per-flow risk-score filter unavailable: this benchmark contains no joined per-flow prediction scores.</span>
      </form>
      {rows.error?<div className="flow-error"><strong>Full flow store unavailable or inconsistent</strong><span>{rows.error}. No smaller flow artifact is substituted.</span></div>:rows.loading?<div className="flow-empty">Loading verified full-benchmark records…</div>:<>
        <div className="flow-result-head"><b>{formatNumber(rows.data?.total)} matching flows</b><span>{rows.data?.source} · {formatNumber(rows.data?.indexed_record_count)} indexed rows in scope</span></div>
        <FlowRows items={rows.data?.items||[]} onInspect={setSelected}/>
        <div className="flow-pagination"><button aria-label="Previous page" disabled={page<=1} onClick={()=>setPage(value=>Math.max(1,value-1))}><ChevronLeft size={15}/> Previous</button><span>Page {page} of {pages}</span><button aria-label="Next page" disabled={page>=pages} onClick={()=>setPage(value=>Math.min(pages,value+1))}>Next page <ChevronRight size={15}/></button></div>
      </>}
    </section>
    <EvidencePanel flow={selected} onClose={()=>setSelected(null)}/>
  </>;
}

export function NetworkTopology({refresh}:{refresh:number}){
  const [dataset,setDataset]=useState<Dataset>('expanded');
  const [selected,setSelected]=useState<{label:string;filters:InvestigationFilter}|null>(null);
  const graph=useJson<{edges:GraphEdge[];scope_note:string}>(`/flows/topology?dataset=${dataset}&limit=100`,refresh);
  const selection=selected?.filters||{};
  const evidence=useJson<FlowPage>(queryString(dataset,selection),refresh,Boolean(selected));
  const built=useMemo(()=>{
    const edges=graph.data?.edges||[];
    const sources=[...new Set(edges.map(edge=>edge.source_ip))];
    const destinations=[...new Set(edges.map(edge=>edge.destination_ip))];
    const services=[...new Set(edges.map(edge=>`${edge.destination_ip}:${edge.destination_port}:${edge.protocol}`))];
    const nodes:Node<NodeData>[]=[
      ...sources.map((ip,index)=>({id:`source:${ip}`,position:{x:25,y:35+index*85},sourcePosition:Position.Right,targetPosition:Position.Left,data:{label:`SOURCE IP\n${ip}`,filters:{source_ip:ip}},style:{width:190,background:'#112638',border:'1px solid #367b98',color:'#d8eff8',fontSize:11}})),
      ...destinations.map((ip,index)=>({id:`destination:${ip}`,position:{x:360,y:35+index*80},sourcePosition:Position.Right,targetPosition:Position.Left,data:{label:`DESTINATION IP\n${ip}`,filters:{destination_ip:ip}},style:{width:190,background:'#172337',border:'1px solid #5866a6',color:'#e2e4ff',fontSize:11}})),
      ...services.map((key,index)=>{const [ip,port,protocol]=key.split(':');const edge=edges.find(item=>`${item.destination_ip}:${item.destination_port}:${item.protocol}`===key)!;return {id:`service:${key}`,position:{x:700,y:35+index*76},targetPosition:Position.Left,sourcePosition:Position.Right,data:{label:`${edge.protocol_name} · ${port}\nSERVICE ON ${ip}`,filters:{destination_ip:ip,destination_port:Number(port)}},style:{width:205,background:'#292319',border:'1px solid #9d7a45',color:'#f3dfb9',fontSize:10}};}),
    ];
    const rfEdges:Edge[]=[];
    for(const edge of edges){
      const suffix=`${edge.source_ip}:${edge.destination_ip}:${edge.destination_port}:${edge.protocol}`;
      rfEdges.push({id:`flow:${suffix}`,source:`source:${edge.source_ip}`,target:`destination:${edge.destination_ip}`,label:`${formatNumber(edge.flow_count)} flows`,type:'smoothstep',animated:false,style:{stroke:'#59b7d1',strokeWidth:Math.min(4,1+Math.log2(edge.flow_count))},labelStyle:{fill:'#b8c8d6',fontSize:9},labelBgStyle:{fill:'#0b1320'}});
      rfEdges.push({id:`service:${suffix}`,source:`destination:${edge.destination_ip}`,target:`service:${edge.destination_ip}:${edge.destination_port}:${edge.protocol}`,label:`${formatNumber(edge.flow_count)} flows`,type:'smoothstep',style:{stroke:'#9b8be4',strokeWidth:1.5},labelStyle:{fill:'#b8c8d6',fontSize:9},labelBgStyle:{fill:'#0b1320'}});
    }
    return {nodes,edges:rfEdges};
  },[graph.data]);
  const selectedRows=evidence.data?.items||[];
  return <>
    <section className="panel topology-panel">
      <div className="panel-head"><div><h2>Observed network topology</h2><span className="tiny-label">TOP 100 SOURCE → DESTINATION → SERVICE AGGREGATES</span></div>
        <select aria-label="Topology dataset scope" value={dataset} onChange={event=>{setDataset(event.target.value as Dataset);setSelected(null);}}><option value="expanded">Expanded benchmark</option><option value="CICIDS2017">CICIDS2017 · full week</option><option value="CICDDoS2019">CIC-DDoS2019 · extension</option></select>
      </div>
      <p className="flow-intro">Nodes and edges are built only from observed raw-flow records. Click a source, destination, or service node to retrieve its matching flow evidence below. This is historical benchmark traffic, not a live network map.</p>
      {graph.error?<div className="flow-error"><strong>Full-benchmark topology unavailable</strong><span>{graph.error}. No smaller artifact is used as a substitute.</span></div>:graph.loading?<div className="flow-empty">Aggregating verified flow topology…</div>:<>
        <div className="topology-legend"><span><i className="topology-source"/>Source IP</span><span><i className="topology-destination"/>Destination IP</span><span><i className="topology-service"/>Destination service</span><span>{graph.data?.edges.length||0} observed edge groups</span></div>
        <div className="topology-canvas" aria-label="Observed flow topology graph"><ReactFlow nodes={built.nodes} edges={built.edges} fitView fitViewOptions={{padding:.16}} nodesDraggable={false} nodesConnectable={false} onNodeClick={(_,node)=>setSelected({label:node.data.label,filters:node.data.filters})} proOptions={{hideAttribution:true}}>
          <Background color="#243348" gap={24}/><Controls/><MiniMap pannable zoomable nodeColor={node=>node.id.startsWith('source:')?'#3284a0':node.id.startsWith('destination:')?'#7776c4':'#ad8749'} />
        </ReactFlow></div>
        <p className="chart-foot">{graph.data?.scope_note} The displayed graph is an aggregated top-100 view; query the flow explorer for unbounded server-side filtering and pagination.</p>
      </>}
    </section>
    {selected&&<section className="panel flow-evidence-panel"><div className="panel-head"><div><h2>Flow evidence</h2><span className="tiny-label">SELECTED OBSERVED NODE · {selected.label}</span></div><button className="text-button" onClick={()=>setSelected(null)}>Clear selection</button></div>
      {evidence.error?<div className="flow-error">{evidence.error}</div>:evidence.loading?<div className="flow-empty">Retrieving matching records…</div>:<><div className="flow-result-head"><b>{formatNumber(evidence.data?.total)} matching flow records</b><span>Maximum 10 rows returned</span></div><FlowRows items={selectedRows} compact/></>}
      <div className="callout"><AlertTriangle size={15}/><span>Full-benchmark per-flow predictions and findings are not present in the saved artifacts. The saved ML run reports aggregated host/minute windows; no per-flow score is inferred or fabricated.</span></div>
    </section>}
  </>;
}
