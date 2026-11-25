// sbnd-h5maker 0.6 (fm4npp #16): step-2 cluster split + post-split point-cloud tree dump, for sbnd-h5maker 0.6.
// = wct-pr.jsonnet (WCT 0.38.0) with a truncated PR pipeline (the visitors that split/rename clusters) and the
// TensorFileSink writing the post-pipeline pctree (blobs + their step-1 truth scalars, in step-2 cluster numbering).
//   wire-cell -c wct-blobdump.jsonnet --tla-str input=qlpctree.tar.gz --tla-str reality=sim \
//       --tla-str out=blobs.tar.gz [--tla-code stages='[...]']
function(input='qlpctree.tar.gz', output_dir='.', reality='sim', out='blobs.tar.gz',
         evt_subdir='pr_evt{run}_{subrun}_{event}',
         stages=['switch_scope', 'unmerge_bundle', 'unmerge_assoc', 'steiner', 'fiducialutils',
                 'tagger_check_tgm', 'tagger_check_stm', 'tagger_check_fc', 'protect_bundle'])

local g = import 'pgraph.jsonnet';
local wc = import 'wirecell.jsonnet';
local tools_maker = import 'pgrapher/common/tools.jsonnet';
local params = import 'pgrapher/experiment/sbnd/simparams.jsonnet';
local tools_all = tools_maker(params);
local tools = tools_all {anodes: [tools_all.anodes[n] for n in [0, 1]]};
local clus = import 'pgrapher/experiment/sbnd/clus.jsonnet';
local clus_maker = clus(output_dir=output_dir, rse_from_ident=true, rse_from_metadata=true,
                        event_from_ident=evt_subdir != '', reality=reality, evt_subdir=evt_subdir);
local pr_stage = import 'pgrapher/experiment/sbnd/sbnd-pr-stage.jsonnet';
local pds = (import 'pgrapher/experiment/sbnd/particle_dataset.jsonnet')();

local source = g.pnode({type: 'TensorFileSource', name: 'ql_pctree', data: {inname: input, prefix: 'clustering_'}},
                       nin=0, nout=1);
local bee = {type: 'BeeSink', name: 'mabc_pr', data: {outname: 'trash-bee.zip', initial_index: 0}};
local pr_node = clus_maker.pr(tools.anodes, dump=false, bee_sink=bee, pipeline_names=stages,
                              particle_dataset=pds.particle_dataset, extra_uses=pds.all,
                              beam_window=pr_stage.beam_window);
local sink = g.pnode({type: 'TensorFileSink', name: 'pr',
                      data: {outname: out, prefix: 'clustering_', dump_mode: false}}, nin=1, nout=0);
local graph = g.intern(innodes=[source], centernodes=[pr_node], outnodes=[sink],
                       edges=[g.edge(source, pr_node, 0, 0), g.edge(pr_node, sink, 0, 0)]);
local app = {type: 'Pgrapher', data: {edges: g.edges(graph)}};
local cmdline = {type: 'wire-cell',
                 data: {plugins: ['WireCellGen', 'WireCellPgraph', 'WireCellAux', 'WireCellSio', 'WireCellSigProc',
                                  'WireCellImg', 'WireCellClus', 'WireCellRoot'],
                        apps: ['Pgrapher']}};
[cmdline, bee] + g.uses(graph) + [app]
