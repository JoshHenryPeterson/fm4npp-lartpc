// SBND reco1 (art ROOT) -> flat per-event vectors, read with gallery.  JIT-compiled by sbnd_h5maker.py.
// Port of the parts of uboone/hdf5maker (branch opensamples, HDF5Maker_module.cc) that qlfm uses:
// event truth, gaushit hits, BackTracker hit->particle energy (edep), particle table with the
// HDF5Maker particle category, PMT ophits, opflashes and their per-channel PE.
// SBND differences: two TPCs (hit tpc), two flash collections (opflashtpc0/1), and simulated cosmics
// (CORSIKA): every particle carries its MCTruth origin (1 beam nu, 2 cosmic, 0 unknown).
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "canvas/Persistency/Common/FindMany.h"
#include "canvas/Persistency/Common/FindManyP.h"
#include "canvas/Utilities/InputTag.h"
#include "gallery/Event.h"
#include "lardataobj/AnalysisBase/BackTrackerMatchingData.h"
#include "lardataobj/MCBase/MCShower.h"
#include "lardataobj/MCBase/MCTrack.h"
#include "lardataobj/RecoBase/Hit.h"
#include "lardataobj/RecoBase/OpFlash.h"
#include "lardataobj/RecoBase/OpHit.h"
#include "lardataobj/Simulation/BeamGateInfo.h"
#include "lardataobj/Simulation/GeneratedParticleInfo.h"
#include "lardataobj/Simulation/ParticleAncestryMap.h"
#include "lardataobj/Simulation/SimChannel.h"
#include "lardataobj/Simulation/SimEnergyDeposit.h"
#include "TFile.h"
#include "TH3F.h"
#include "nusimdata/SimulationBase/MCParticle.h"
#include "nusimdata/SimulationBase/MCTruth.h"

namespace sbndh5 {

struct Labels {
  std::string hit = "gaushit";
  std::string hit_truth = "gaushitTruthMatch";
  std::string truth = "generator";
  std::string mcparticle = "largeant";
  std::string mcreco = "mcreco";
  std::string ophit = "ophitpmt";
  std::vector<std::string> opflash = {"opflashtpc0", "opflashtpc1"};
  std::string beamgate = "generator";
  // Hit truth.  The reco1 gaushitTruthMatch assns are empty in the Gen2 MC, so by default the hit ->
  // particle energy is rebuilt from the SimChannels as larsim BackTracker::HitToTrackIDEs does: IDEs on the
  // hit channel with TDC in [PeakTime - RMS, PeakTime + RMS] + tdc_offset (TPCTick2TDC; SBND MC:
  // (TriggerTime 1700 us + TriggerOffsetTPC -205 us) / 0.5 us = 2990).
  std::string simchannel = "simtpc2d:simpleSC";
  std::string ancestry = "largeant";
  double tdc_offset = 2990;
  double hit_time_rms = 1.0;
  // Fallback when neither the assns nor the SimChannels carry deposits (the Gen2 MC): project the Geant4
  // deposits (before SCE) through the forward SCE map and the drift onto (channel, tick); see Geo.
  std::string sed = "ionandscint:priorSCE";
};

// Geometry for the deposit projection.  Six wire planes (tpc*3 + plane), each linear:
// channel = channel0 + channel_step * round(((y, z) - wire0_yz) . pitch_dir / pitch).
// tick = trigger_tick + (t_G4[us] + |x_plane - x_sce| / drift_v) / tick_us + dt[plane]   (G4 t = 0 at the trigger)
struct Geo {
  std::vector<double> x, y0, z0, pdy, pdz, pitch, dt;
  std::vector<int> n, ch0, step;
  double drift_v = 0.1563, tick_us = 0.5, trig_tick = 410, window = 1.0, pad = 0.0;
  double av[6] = {-200, 200, -203.7325, 203.7325, 0, 501};  // active volume x, y, z ranges [cm]
  TH3F* sce[2][3] = {{nullptr, nullptr, nullptr}, {nullptr, nullptr, nullptr}};  // [E(x<0), W(x>0)][x, y, z]
  bool use_sce = false;
};

inline bool load_sce(Geo& g, const std::string& path) {
  TFile* f = TFile::Open(path.c_str(), "READ");
  if (!f || f->IsZombie()) return false;
  const char* side[2] = {"E", "W"};
  const char* comp[3] = {"X", "Y", "Z"};
  for (int s = 0; s < 2; ++s)
    for (int c = 0; c < 3; ++c) {
      auto h = dynamic_cast<TH3F*>(f->Get(Form("TrueFwd_Displacement_%s_%s", comp[c], side[s])));
      if (!h) return false;
      h->SetDirectory(nullptr);
      g.sce[s][c] = h;
    }
  g.use_sce = true;
  return true;
}

inline double sce_interp(TH3F* h, double x, double y, double z) {
  auto clamp = [](const TAxis* a, double v) {
    double lo = a->GetBinCenter(1), hi = a->GetBinCenter(a->GetNbins()), pad = 1e-3 * a->GetBinWidth(1);
    return v < lo + pad ? lo + pad : (v > hi - pad ? hi - pad : v);
  };
  return h->Interpolate(clamp(h->GetXaxis(), x), clamp(h->GetYaxis(), y), clamp(h->GetZaxis(), z));
}

struct Dep { float tick; float e, ne; int tid; };

// deposits per channel, sorted by tick
inline void project_seds(gallery::Event& ev, const Labels& L, const Geo& g,
                         std::map<int, std::vector<Dep>>& bych, long& n_sed) {
  bych.clear();
  auto sh = ev.getValidHandle<std::vector<sim::SimEnergyDeposit>>(art::InputTag(L.sed));
  n_sed = sh->size();
  for (auto const& d : *sh) {
    if (d.NumElectrons() <= 0) continue;
    auto m = d.MidPoint();
    double x = m.X(), y = m.Y(), z = m.Z();
    int tpc = x < 0 ? 0 : 1;
    if (g.use_sce) {
      double dx = sce_interp(g.sce[tpc][0], x, y, z), dy = sce_interp(g.sce[tpc][1], x, y, z),
             dz = sce_interp(g.sce[tpc][2], x, y, z);
      x += dx; y += dy; z += dz;
      tpc = x < 0 ? 0 : 1;
    }
    double t_us = d.Time() * 1e-3;
    for (int p = 0; p < 3; ++p) {
      int k = tpc * 3 + p;
      double dist = (tpc == 0) ? x - g.x[k] : g.x[k] - x;  // >0 inside the drift volume
      if (dist < 0) continue;
      double w = ((y - g.y0[k]) * g.pdy[k] + (z - g.z0[k]) * g.pdz[k]) / g.pitch[k];
      long iw = std::lround(w);
      if (iw < 0 || iw >= g.n[k]) continue;
      int ch = g.ch0[k] + g.step[k] * iw;
      float tick = g.trig_tick + (t_us + dist / g.drift_v) / g.tick_us + g.dt[k];
      bych[ch].push_back({tick, (float)d.Energy(), (float)d.NumElectrons(), d.TrackID()});
    }
  }
  for (auto& kv : bych)
    std::sort(kv.second.begin(), kv.second.end(), [](const Dep& a, const Dep& b) { return a.tick < b.tick; });
}

// HDF5Maker ParticleCategory
enum Cat { Pion = 0, Muon = 1, Kaon = 2, Proton = 3, Electron = 4, Michel = 5, Delta = 6, OtherNu = 7, Photon = 8 };

struct Out {
  int run = 0, subrun = 0, event = 0;
  // event truth: the neutrino chosen by choose_nu() (nu_sel = its nu_index, -1 if none);
  // n_nu = number of generator MCTruths with a neutrino
  int n_nu = 0, nu_sel = -1, is_cc = -1, nu_pdg = 0, nu_mode = -1, nu_int_type = -1, lep_pdg = 0;
  float nu_energy = -1, lep_energy = -1, nu_time = 0;
  std::vector<float> nu_dir, nu_vtx;
  // nu_table: one row per generator MCTruth with a neutrino; nu_index = position in the generator collection
  std::vector<int> v_index, v_pdg, v_cc, v_mode, v_int, v_lep_pdg, v_in_av;
  std::vector<float> v_energy, v_lep_energy, v_time, v_vtx, v_dir, v_ecoll, v_nhits_coll;
  float beam_gate_start = -1, beam_gate_width = -1;
  // hit_table
  std::vector<int> h_tpc, h_plane, h_wire, h_channel;
  std::vector<float> h_time, h_integral, h_rms, h_amp;
  // edep (hit -> particle)
  std::vector<int> e_hit, e_g4, e_origin, e_maxide, e_nu;
  std::vector<float> e_energy, e_frac, e_nelec;
  // particle_table
  std::vector<int> p_g4, p_pdg, p_parent, p_cat, p_inst, p_origin, p_nu;
  std::vector<float> p_mom, p_start, p_end, p_t;
  std::vector<std::string> p_sproc, p_eproc;
  // ophit_table
  std::vector<int> o_ch;
  std::vector<float> o_t, o_tabs, o_width, o_area, o_amp, o_pe;
  // opflash_table, opflashsumpe_table
  std::vector<int> f_id, f_tpc;
  std::vector<float> f_time, f_twidth, f_y, f_yw, f_z, f_zw, f_pe;
  std::vector<int> s_id, s_flash, s_ch;
  std::vector<float> s_pe;
  int n_mcp = 0, n_mcp_no_truth = 0, n_ide_unmapped = 0;
  int truth_source = 0;  // 1 = gaushitTruthMatch assns, 2 = SimChannel backtracking, 3 = deposit projection
  long n_sed = 0;
  double sed_e_nu = 0, sed_e_nu_matched = 0;   // projected nu deposit energy (collection plane), and on hits
  long n_assns_hit_truth = -1, n_assns_mcp_truth = -1;
};

struct BtPart {
  int category;
  std::vector<unsigned int> tids;
  int nhits = 0;
};

inline void fill(gallery::Event& ev, Out& o, const Labels& L, const Geo& G) {
  o = Out();
  auto const& aux = ev.eventAuxiliary();
  o.run = aux.run();
  o.subrun = aux.subRun();
  o.event = aux.event();

  // ---- neutrino truth (HDF5Maker EventInfo "nu")
  o.nu_dir.assign(3, 0.f);
  o.nu_vtx.assign(3, -9999.f);
  std::map<size_t, size_t> nurow;  // generator MCTruth index -> nu_table row
  {
    auto th = ev.getValidHandle<std::vector<simb::MCTruth>>(art::InputTag(L.truth));
    for (size_t k = 0; k < th->size(); ++k) {
      auto const& t = (*th)[k];
      if (!t.NeutrinoSet()) continue;
      ++o.n_nu;
      auto const& nu = t.GetNeutrino();
      auto const& p = nu.Nu();
      nurow[k] = o.v_index.size();
      o.v_index.push_back(k);
      o.v_pdg.push_back(p.PdgCode());
      o.v_cc.push_back(nu.CCNC() == simb::kCC);
      o.v_mode.push_back(nu.Mode());
      o.v_int.push_back(nu.InteractionType());
      o.v_lep_pdg.push_back(nu.Lepton().PdgCode());
      o.v_energy.push_back(p.E());
      o.v_lep_energy.push_back(nu.Lepton().E());
      o.v_time.push_back(p.T());
      o.v_vtx.insert(o.v_vtx.end(), {(float)p.Vx(), (float)p.Vy(), (float)p.Vz()});
      auto d = p.Momentum().Vect().Unit();
      o.v_dir.insert(o.v_dir.end(), {(float)d.X(), (float)d.Y(), (float)d.Z()});
      o.v_in_av.push_back(p.Vx() > G.av[0] && p.Vx() < G.av[1] && p.Vy() > G.av[2] && p.Vy() < G.av[3] &&
                          p.Vz() > G.av[4] && p.Vz() < G.av[5]);
      o.v_ecoll.push_back(0.f);
      o.v_nhits_coll.push_back(0.f);
    }
  }
  {
    gallery::Handle<std::vector<sim::BeamGateInfo>> bg;
    if (ev.getByLabel(art::InputTag(L.beamgate), bg) && !bg->empty()) {
      o.beam_gate_start = bg->front().Start();
      o.beam_gate_width = bg->front().Width();
    }
  }

  // ---- MC particles and their origin (MCParticle -> MCTruth assns written by largeant)
  auto mcp = ev.getValidHandle<std::vector<simb::MCParticle>>(art::InputTag(L.mcparticle));
  std::map<int, size_t> tid2idx;
  std::vector<int> origin(mcp->size(), 0), pnu(mcp->size(), -1);
  {
    art::FindManyP<simb::MCTruth, sim::GeneratedParticleInfo> fmt(mcp, ev, art::InputTag(L.mcparticle));
    for (size_t i = 0; i < mcp->size(); ++i) {
      tid2idx[(*mcp)[i].TrackId()] = i;
      auto const& v = fmt.at(i);
      if (v.empty()) { ++o.n_mcp_no_truth; continue; }
      origin[i] = v[0]->Origin();
      // beam neutrinos come only from the generator collection: the Ptr key is the MCTruth index there
      if (origin[i] == simb::kBeamNeutrino && nurow.count(v[0].key())) pnu[i] = v[0].key();
    }
  }
  o.n_mcp = mcp->size();
  // particles dropped by largeant (larg4 refactor): descendant -> kept ancestor
  std::map<int, int> dropped2anc;
  {
    gallery::Handle<std::vector<sim::ParticleAncestryMap>> am;
    if (ev.getByLabel(art::InputTag(L.ancestry), am))
      for (auto const& m : *am)
        for (auto const& kv : m.GetMap())
          for (int d : kv.second) dropped2anc[std::abs(d)] = std::abs(kv.first);
  }
  // track id as stored (IDE or particle) -> kept MCParticle track id (0 = unknown)
  auto kept_tid = [&](int tid) {
    int a = std::abs(tid);
    if (tid2idx.count(a)) return a;
    auto it = dropped2anc.find(a);
    return (it != dropped2anc.end() && tid2idx.count(it->second)) ? it->second : 0;
  };
  auto origin_of = [&](int tid) {
    auto it = tid2idx.find(kept_tid(tid));
    return it == tid2idx.end() ? 0 : origin[it->second];
  };
  auto nu_of = [&](int tid) {
    auto it = tid2idx.find(kept_tid(tid));
    return it == tid2idx.end() ? -1 : pnu[it->second];
  };

  // ---- hits + BackTracker matching
  {
    gallery::Handle<art::Assns<recob::Hit, simb::MCParticle, anab::BackTrackerHitMatchingData>> ah;
    if (ev.getByLabel(art::InputTag(L.hit_truth), ah)) o.n_assns_hit_truth = ah->size();
    gallery::Handle<art::Assns<simb::MCParticle, simb::MCTruth, sim::GeneratedParticleInfo>> am;
    if (ev.getByLabel(art::InputTag(L.mcparticle), am)) o.n_assns_mcp_truth = am->size();
  }
  auto hits = ev.getValidHandle<std::vector<recob::Hit>>(art::InputTag(L.hit));
  std::set<int> g4id;
  size_t nh = hits->size();
  // per hit: (kept track id, energy, fraction, is max, electrons)
  struct M { int tid; float e, frac, ne; int ismax; };
  std::vector<std::vector<M>> match(nh);
  if (o.n_assns_hit_truth > 0) {
    o.truth_source = 1;
    art::FindMany<simb::MCParticle, anab::BackTrackerHitMatchingData> fmh(hits, ev, art::InputTag(L.hit_truth));
    for (size_t ih = 0; ih < nh; ++ih) {
      auto const& parts = fmh.at(ih);
      auto const& data = fmh.data(ih);
      for (size_t k = 0; k < parts.size(); ++k)
        match[ih].push_back({parts[k]->TrackId(), (float)data[k]->energy, (float)data[k]->ideFraction,
                             (float)data[k]->numElectrons, (int)data[k]->isMaxIDE});
    }
  }
  bool sc_has_ides = false;
  if (o.truth_source == 0) {
    gallery::Handle<std::vector<sim::SimChannel>> sch;
    if (ev.getByLabel(art::InputTag(L.simchannel), sch))
      for (auto const& sc : *sch) if (!sc.TDCIDEMap().empty()) { sc_has_ides = true; break; }
  }
  if (o.truth_source == 0 && !sc_has_ides) {
    o.truth_source = 3;
    std::map<int, std::vector<Dep>> bych;
    project_seds(ev, L, G, bych, o.n_sed);
    // nu deposit energy on the collection planes, for the matched fraction
    for (auto const& kv : bych) {
      int ch = kv.first;
      bool coll = (ch >= G.ch0[2] - (G.step[2] < 0 ? G.n[2] - 1 : 0) && ch <= G.ch0[2] + (G.step[2] > 0 ? G.n[2] - 1 : 0)) ||
                  (ch >= G.ch0[5] - (G.step[5] < 0 ? G.n[5] - 1 : 0) && ch <= G.ch0[5] + (G.step[5] > 0 ? G.n[5] - 1 : 0));
      if (!coll) continue;
      for (auto const& d : kv.second) if (origin_of(d.tid) == 1) o.sed_e_nu += d.e;
    }
    for (size_t ih = 0; ih < nh; ++ih) {
      auto const& h = (*hits)[ih];
      auto it = bych.find(h.Channel());
      if (it == bych.end()) continue;
      double t0 = h.PeakTime() - G.window * h.RMS() - G.pad, t1 = h.PeakTime() + G.window * h.RMS() + G.pad;
      auto const& v = it->second;
      auto lo = std::lower_bound(v.begin(), v.end(), t0, [](const Dep& a, double t) { return a.tick < t; });
      std::map<int, std::pair<double, double>> acc;
      double etot = 0;
      for (auto d = lo; d != v.end() && d->tick <= t1; ++d) {
        int k = kept_tid(d->tid);
        if (k == 0) { ++o.n_ide_unmapped; k = -std::abs(d->tid); }
        acc[k].first += d->e;
        acc[k].second += d->ne;
        etot += d->e;
      }
      double emax = 0;
      for (auto const& kv : acc) emax = std::max(emax, kv.second.first);
      for (auto const& kv : acc) {
        match[ih].push_back({kv.first, (float)kv.second.first, (float)(etot > 0 ? kv.second.first / etot : 0),
                             (float)kv.second.second, (int)(kv.second.first == emax)});
        if (h.WireID().Plane == 2 && kv.first > 0 && origin_of(kv.first) == 1) o.sed_e_nu_matched += kv.second.first;
      }
    }
  } else if (o.truth_source == 0) {
    o.truth_source = 2;
    auto sch = ev.getValidHandle<std::vector<sim::SimChannel>>(art::InputTag(L.simchannel));
    std::map<unsigned int, const sim::SimChannel*> bych;
    for (auto const& sc : *sch) bych[sc.Channel()] = &sc;
    for (size_t ih = 0; ih < nh; ++ih) {
      auto const& h = (*hits)[ih];
      auto it = bych.find(h.Channel());
      if (it == bych.end()) continue;
      double t0 = h.PeakTime() - L.hit_time_rms * h.RMS() + L.tdc_offset;
      double t1 = h.PeakTime() + L.hit_time_rms * h.RMS() + L.tdc_offset;
      if (t0 < 0) t0 = 0;
      if (t1 < 0) continue;
      // merge IDEs of dropped particles into their kept ancestor
      std::map<int, std::pair<double, double>> acc;  // kept tid -> (energy, electrons)
      double etot = 0;
      for (auto const& ide : it->second->TrackIDEs((sim::SimChannel::TDC_t)t0, (sim::SimChannel::TDC_t)t1)) {
        int k = kept_tid(ide.trackID);
        if (k == 0) { ++o.n_ide_unmapped; k = -std::abs(ide.trackID); }
        acc[k].first += ide.energy;
        acc[k].second += ide.numElectrons;
        etot += ide.energy;
      }
      double emax = 0;
      for (auto const& kv : acc) emax = std::max(emax, kv.second.first);
      for (auto const& kv : acc)
        match[ih].push_back({kv.first, (float)kv.second.first, (float)(etot > 0 ? kv.second.first / etot : 0),
                             (float)kv.second.second, (int)(kv.second.first == emax)});
    }
  }
  for (size_t ih = 0; ih < nh; ++ih) {
    auto const& h = (*hits)[ih];
    auto const& w = h.WireID();
    o.h_tpc.push_back(w.TPC);
    o.h_plane.push_back(w.Plane);
    o.h_wire.push_back(w.Wire);
    o.h_channel.push_back(h.Channel());
    o.h_time.push_back(h.PeakTime());
    o.h_integral.push_back(h.Integral());
    o.h_rms.push_back(h.RMS());
    o.h_amp.push_back(h.PeakAmplitude());
    for (auto const& m : match[ih]) {
      if (m.tid > 0) g4id.insert(m.tid);
      o.e_hit.push_back(ih);
      o.e_g4.push_back(m.tid);
      o.e_origin.push_back(m.tid > 0 ? origin_of(m.tid) : 0);
      int inu = m.tid > 0 ? nu_of(m.tid) : -1;
      o.e_nu.push_back(inu);
      if (inu >= 0 && h.WireID().Plane == 2) {
        size_t r = nurow[inu];
        o.v_ecoll[r] += m.e;
        if (m.ismax) o.v_nhits_coll[r] += 1;
      }
      o.e_energy.push_back(m.e);
      o.e_frac.push_back(m.frac);
      o.e_maxide.push_back(m.ismax);
      o.e_nelec.push_back(m.ne);
    }
  }

  // ---- event-level neutrino (choose_nu): the one with the most deposited energy on collection-plane hits;
  // ties (e.g. none visible) go to the one with the vertex in the active volume, then to the lowest nu_index.
  {
    int best = -1;
    for (size_t r = 0; r < o.v_index.size(); ++r) {
      if (best < 0) { best = r; continue; }
      if (o.v_ecoll[r] > o.v_ecoll[best] ||
          (o.v_ecoll[r] == o.v_ecoll[best] && o.v_in_av[r] > o.v_in_av[best])) best = r;
    }
    if (best >= 0) {
      o.nu_sel = o.v_index[best];
      o.is_cc = o.v_cc[best];
      o.nu_pdg = o.v_pdg[best];
      o.nu_energy = o.v_energy[best];
      o.lep_energy = o.v_lep_energy[best];
      o.lep_pdg = o.v_lep_pdg[best];
      o.nu_mode = o.v_mode[best];
      o.nu_int_type = o.v_int[best];
      o.nu_time = o.v_time[best];
      o.nu_vtx = {o.v_vtx[3 * best], o.v_vtx[3 * best + 1], o.v_vtx[3 * best + 2]};
      o.nu_dir = {o.v_dir[3 * best], o.v_dir[3 * best + 1], o.v_dir[3 * best + 2]};
    }
  }

  // ---- HDF5Maker particle categories (initBacktrackingParticleVec, nhitcut 5)
  std::vector<BtPart> bt;
  {
    gallery::Handle<std::vector<sim::MCShower>> shw;
    gallery::Handle<std::vector<sim::MCTrack>> trk;
    if (ev.getByLabel(art::InputTag(L.mcreco), shw)) {
      for (auto const& s : *shw) {
        int c = -1;
        if (s.Process() == "primary" && s.PdgCode() == 22) c = Photon;
        else if (s.MotherPdgCode() == 111 && s.Process() == "Decay" && s.MotherProcess() == "primary") c = Photon;
        else if (s.Process() == "primary" && std::abs(s.PdgCode()) == 11) c = Electron;
        else if (std::abs(s.MotherPdgCode()) == 13 && s.Process() == "Decay") c = Michel;
        else if (std::abs(s.MotherPdgCode()) == 13 && s.Process() == "muIoni" && s.Start().Momentum().P() > 10) c = Delta;
        else if (std::abs(s.MotherPdgCode()) == 211 && s.Process() == "hIoni" && s.Start().Momentum().P() > 10) c = Delta;
        else continue;
        bt.push_back({c, s.DaughterTrackID(), 0});
      }
    }
    if (ev.getByLabel(art::InputTag(L.mcreco), trk)) {
      for (auto const& t : *trk) {
        int c = -1, a = std::abs(t.PdgCode());
        if (a == 13) c = Muon;
        else if (a == 2212) c = Proton;
        else if (a == 211) c = Pion;
        else if (a == 321) c = Kaon;
        else continue;
        bt.push_back({c, {t.TrackID()}, 0});
      }
    }
    for (size_t ih = 0; ih < nh; ++ih) {
      for (auto const& m : match[ih]) {
        if (m.ismax != 1 || m.tid <= 0) continue;
        unsigned int tid = m.tid;
        for (auto& b : bt)
          if (std::find(b.tids.begin(), b.tids.end(), tid) != b.tids.end()) b.nhits++;
      }
    }
    std::vector<BtPart> keep;
    for (auto& b : bt) if (b.nhits > 5) keep.push_back(b);
    bt.swap(keep);
  }

  // ---- particle table: particles with deposits + their ancestors
  std::set<int> all = g4id;
  for (int id : g4id) {
    auto it = tid2idx.find(std::abs(id));
    while (it != tid2idx.end()) {
      int m = (*mcp)[it->second].Mother();
      if (m == 0 || all.count(std::abs(m))) break;
      all.insert(std::abs(m));
      it = tid2idx.find(std::abs(m));
    }
  }
  for (int id : all) {
    auto it = tid2idx.find(std::abs(id));
    if (it == tid2idx.end()) continue;
    auto const& p = (*mcp)[it->second];
    int cat = OtherNu, inst = -1, ndelta = 0;
    for (size_t i = 0; i < bt.size(); ++i) {
      if (bt[i].category == Delta) ndelta++;
      if (std::find(bt[i].tids.begin(), bt[i].tids.end(), (unsigned)std::abs(id)) == bt[i].tids.end()) continue;
      cat = bt[i].category;
      if (cat != Delta) inst = i - ndelta;
      break;
    }
    if (cat == OtherNu && inst == -1 && std::abs(p.PdgCode()) == 11 && (p.Process() == "muIoni" || p.Process() == "hIoni")) {
      ndelta = 0;
      for (size_t i = 0; i < bt.size(); ++i) {
        if (bt[i].category == Delta) ndelta++;
        if ((int)bt[i].tids[0] != p.Mother()) continue;
        cat = bt[i].category;
        inst = i - ndelta;
        break;
      }
    }
    o.p_g4.push_back(std::abs(id));
    o.p_pdg.push_back(p.PdgCode());
    o.p_parent.push_back(p.Mother());
    o.p_cat.push_back(cat);
    o.p_inst.push_back(inst);
    o.p_origin.push_back(origin[it->second]);
    o.p_nu.push_back(pnu[it->second]);
    o.p_mom.push_back(p.P());
    o.p_start.insert(o.p_start.end(), {(float)p.Vx(), (float)p.Vy(), (float)p.Vz()});
    o.p_end.insert(o.p_end.end(), {(float)p.EndX(), (float)p.EndY(), (float)p.EndZ()});
    o.p_t.push_back(p.T());
    o.p_sproc.push_back(p.Process());
    o.p_eproc.push_back(p.EndProcess());
  }

  // ---- optical hits (PMTs)
  {
    gallery::Handle<std::vector<recob::OpHit>> oh;
    if (ev.getByLabel(art::InputTag(L.ophit), oh)) {
      for (auto const& h : *oh) {
        o.o_ch.push_back(h.OpChannel());
        o.o_t.push_back(h.PeakTime());
        o.o_tabs.push_back(h.PeakTimeAbs());
        o.o_width.push_back(h.Width());
        o.o_area.push_back(h.Area());
        o.o_amp.push_back(h.Amplitude());
        o.o_pe.push_back(h.PE());
      }
    }
  }
  // ---- flashes (both TPCs, one running flash_id)
  int fid = 0, sid = 0;
  for (size_t itpc = 0; itpc < L.opflash.size(); ++itpc) {
    gallery::Handle<std::vector<recob::OpFlash>> fh;
    if (!ev.getByLabel(art::InputTag(L.opflash[itpc]), fh)) continue;
    for (auto const& f : *fh) {
      o.f_id.push_back(fid);
      o.f_tpc.push_back(itpc);
      o.f_time.push_back(f.Time());
      o.f_twidth.push_back(f.TimeWidth());
      o.f_y.push_back(f.YCenter());
      o.f_yw.push_back(f.YWidth());
      o.f_z.push_back(f.ZCenter());
      o.f_zw.push_back(f.ZWidth());
      o.f_pe.push_back(f.TotalPE());
      auto const& pes = f.PEs();
      for (size_t ch = 0; ch < pes.size(); ++ch) {
        if (pes[ch] <= 0.) continue;
        o.s_id.push_back(sid++);
        o.s_flash.push_back(fid);
        o.s_ch.push_back(ch);
        o.s_pe.push_back(pes[ch]);
      }
      ++fid;
    }
  }
}

// Fraction-of-hits-with-IDEs vs tdc offset, to pin the tick -> TDC convention on real events.
inline std::vector<long> scan_tdc(gallery::Event& ev, const Labels& L, const std::vector<double>& offs) {
  auto hits = ev.getValidHandle<std::vector<recob::Hit>>(art::InputTag(L.hit));
  auto sch = ev.getValidHandle<std::vector<sim::SimChannel>>(art::InputTag(L.simchannel));
  std::map<unsigned int, const sim::SimChannel*> bych;
  for (auto const& sc : *sch) bych[sc.Channel()] = &sc;
  std::vector<long> n(offs.size(), 0);
  for (auto const& h : *hits) {
    auto it = bych.find(h.Channel());
    if (it == bych.end()) continue;
    for (size_t i = 0; i < offs.size(); ++i) {
      double t0 = h.PeakTime() - h.RMS() + offs[i], t1 = h.PeakTime() + h.RMS() + offs[i];
      if (t1 < 0) continue;
      if (!it->second->TrackIDEs((sim::SimChannel::TDC_t)std::max(0., t0), (sim::SimChannel::TDC_t)t1).empty()) ++n[i];
    }
  }
  return n;
}

// SimChannel summary: n channels, channel min/max, TDC min/max, total IDEs, sum energy
inline std::vector<double> sc_summary(gallery::Event& ev, const Labels& L) {
  std::vector<double> r(6, 0);
  gallery::Handle<std::vector<sim::SimChannel>> sch;
  if (!ev.getByLabel(art::InputTag(L.simchannel), sch)) { r[0] = -1; return r; }
  r[0] = sch->size(); r[1] = 1e9; r[2] = -1; r[3] = 1e9; r[4] = -1e9;
  double nide = 0, e = 0;
  for (auto const& sc : *sch) {
    r[1] = std::min<double>(r[1], sc.Channel()); r[2] = std::max<double>(r[2], sc.Channel());
    for (auto const& t : sc.TDCIDEMap()) {
      r[3] = std::min<double>(r[3], t.first); r[4] = std::max<double>(r[4], t.first);
      for (auto const& ide : t.second) { nide += 1; e += ide.energy; }
    }
  }
  r.push_back(nide); r.push_back(e);
  return r;
}

// Hits with >= 1 projected deposit, per plane (0..2), for each common time offset added to G.dt;
// returns [offset index][plane] flattened.  Also reports channel consistency: hits whose (tpc, plane) disagree
// with the plane of the model channel are counted in the last 3 entries (per plane) for offset 0.
inline std::vector<long> scan_proj(gallery::Event& ev, const Labels& L, const Geo& G, const std::vector<double>& offs) {
  auto hits = ev.getValidHandle<std::vector<recob::Hit>>(art::InputTag(L.hit));
  std::map<int, std::vector<Dep>> bych;
  long nsed = 0;
  project_seds(ev, L, G, bych, nsed);
  std::vector<long> n(offs.size() * 3 + 3, 0);
  for (auto const& h : *hits) {
    auto it = bych.find(h.Channel());
    if (it == bych.end()) continue;
    auto const& v = it->second;
    int p = h.WireID().Plane;
    for (size_t i = 0; i < offs.size(); ++i) {
      double t0 = h.PeakTime() - G.window * h.RMS() - offs[i], t1 = h.PeakTime() + G.window * h.RMS() - offs[i];
      auto lo = std::lower_bound(v.begin(), v.end(), t0, [](const Dep& a, double t) { return a.tick < t; });
      if (lo != v.end() && lo->tick <= t1) ++n[i * 3 + p];
    }
  }
  // channel model check: a hit's channel must belong to the model plane (tpc*3+plane)
  for (auto const& h : *hits) {
    int k = h.WireID().TPC * 3 + h.WireID().Plane;
    int lo = G.step[k] > 0 ? G.ch0[k] : G.ch0[k] - G.n[k] + 1, hi = lo + G.n[k] - 1;
    if ((int)h.Channel() < lo || (int)h.Channel() > hi) ++n[offs.size() * 3 + h.WireID().Plane];
  }
  return n;
}

// copy helpers for numpy (cppyy passes a numpy buffer as a plain pointer; no templates: cppyy cannot deduce them)
inline void copy_i(const std::vector<int>& v, int* out) { std::copy(v.begin(), v.end(), out); }
inline void copy_f(const std::vector<float>& v, float* out) { std::copy(v.begin(), v.end(), out); }

}  // namespace sbndh5
