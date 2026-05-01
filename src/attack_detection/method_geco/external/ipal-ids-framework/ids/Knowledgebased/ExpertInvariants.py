from ids.featureids import MetaIDS
import ipal_iids.settings as settings
import json

def _inv(a, b):
    fulfilled = not a or b  # a => b
    return not fulfilled  # raise alert


class ExpertInvariants(MetaIDS):
    _name = "ExpertInvariants"
    _description = "TODO"
    _requires = ["live.state"]
    _expert_default_settings = { }

    def __init__(self, name=None):
        super().__init__(name=name)
        self._add_default_settings(self._expert_default_settings)

    def train(self, ipal=None, state=None):
        pass  # Nothing to train here

    def new_state_msg(self, msg):
        s = msg["state"]
        alert = False

        ### Define predicats
        ait201gt260 = s["AIT201"] > 260
        ait201lt250 = s["AIT201"] < 250
        ait202geq705 = s["AIT202"] >= 7.05
        ait202lt695 = s["AIT202"] < 6.95
        ait203gt500 = s["AIT203"] > 500
        ait203leq420 = s["AIT203"] <= 420

        mv101closed = s["MV101"] == 1
        mv101open = s["MV101"] == 2
        mv201closed = s["MV201"] == 1
        mv201open = s["MV201"] == 2
        mv301closed = s["MV301"] == 1
        mv301open = s["MV301"] == 2
        mv302closed = s["MV302"] == 1
        mv302open = s["MV302"] == 2

        p101off = s["P101"] == 1
        p101on = s["P101"] == 2
        p102off = s["P102"] == 1
        p102on = s["P102"] == 2
        p201off = s["P201"] == 1
        p201on = s["P201"] == 2
        p202off = s["P202"] == 1
        p202on = s["P202"] == 2
        p203off = s["P203"] == 1
        p203on = s["P203"] == 2
        p204off = s["P204"] == 1
        p204on = s["P204"] == 2
        p205off = s["P205"] == 1
        p205on = s["P205"] == 2
        p206off = s["P206"] == 1
        p206on = s["P206"] == 2
        p301off = s["P301"] == 1
        p301on = s["P301"] == 2
        p302off = s["P302"] == 1
        p302on = s["P302"] == 2
        p401off = s["P401"] == 1
        p401on = s["P401"] == 2
        p402off = s["P402"] == 1
        p402on = s["P402"] == 2
        p403off = s["P403"] == 1
        p403on = s["P403"] == 2
        p404off = s["P404"] == 1
        p405on = s["P404"] == 2
        p501off = s["P501"] == 1
        p501on = s["P501"] == 2
        p601off = s["P601"] == 1
        p601on = s["P601"] == 2

        uv401off = s["UV401"] == 1
        uv401on = s["UV401"] == 2

        ait402h = s["AIT402"] > 330 # estimated
        ait402l = s["AIT402"] < 160 # estimated
        ait503h = s["AIT503"] > 280 # estimated
        fit201ll = s["FIT201"] < 2.0 # estimated
        lit101h = s["LIT101"] > 810 # estimated
        lit101hh = s["LIT101"] > 810 # estimated
        lit101l = s["LIT101"] < 490 # estimated
        lit101ll = s["LIT101"] < 490 #estimated
        lit301h = s["LIT301"] > 1010 # estimated
        lit301l = s["LIT301"] < 790 # estimated
        lit301ll = s["LIT301"] < 785 # estimated
        lit401h = s["LIT401"] > 900 # estimated
        lit401l = s["LIT401"] < 300 # estimated
        fit401ll = s["FIT401"] < 1.0 # estimated

        lit301hh = s["LIT301"] > 1015
        lit401ll = s["LIT401"] < 290

        ### 1 iTrust comparison.pdf

        #  1 LIT101 Low => MV101 open
        alert |= _inv(lit101l, mv101open) # 2,0

        #  2 LIT101 High => MV101 close
        alert |= _inv(lit101h, mv101closed) # 2,0

        #  3 LIT101 <= Low Low => P101 or P102 OFF
        alert |= _inv(lit101ll, p101off or p102off) # 3,0

        #  4 LIT301 Low => P101 or P102 ON
        alert |= _inv(lit301l, p101on or p102on) # 4,0

        #  5 LIT301 High => P101 or P102 OFF
        alert |= _inv(lit301h, p101off or p102off) # 0,0 (3,0 with lower lit301h)

        #  6 LIT301 Low => MV201 Open
        alert |= _inv(lit301l, mv201open) # 2,0

        #  7 LIT301 High => MV201 close
        alert |= _inv(lit301h, mv201closed) # 3,0

        #  8 MV201 Open => P201, P202, P204, P206 ON
        #alert |= _inv(mv201open, p201on and p202on and p204on and p206on) # 24,102

        #  9 FIT201 Low Low => P201, P202, P204, P206 OFF
        alert |= _inv(fit201ll, p201off and p202off and p204off and p206off) # 0,0

        # 10 AIT201 > 260 uS/cm => P201 or P202 OFF
        alert |= _inv(ait201gt260, p201off and p202off) # 0,0

        # 11 AIT201 < 250 uS/cm => P201 or P202 ON
        #alert |= _inv(ait201lt250, p201on or p202on)  # 13,111

        # 12 AIT503 High => P201 or P202 OFF
        alert |= _inv(ait503h, p201off or p202off) # 0,0

        # 13 AIT503 not High => P201 or P202 ON
        #alert |= _inv(not ait503h, p201on or p202on) # 16,145

        # 14 AIT202 < 6.95 => P203 or P204 OFF
        alert |= _inv(ait202lt695, p203off or p204off) # 0,0

        # 15 AIT202 >= 7.05 => P203 or P204 ON
        #alert |= _inv(ait202geq705, p203on or p204on) # 21,106

        # 16 AIT203 > 500 mV => P205 or P206 OFF
        alert |= _inv(ait203gt500, p205off or p206off) # 0,0

        # 17 AIT203 <= 420 mV => P205 or P206 ON
        #alert |= _inv(ait203leq420, p205on or p206on) # 21,106

        # 18 AIT402 High => P205 or P206 OFF
        alert |= _inv(ait402h, p205off or p206off) # 0,0

        # 19 AIT402 not High => P205 or P206 ON
        #alert |= _inv(not ait402h, p205on or p206on) # 26,106

        # 20 LIT301 <= Low Low => P301 or P302 OFF
        alert |= _inv(lit301ll, p301off or p302off) # 0,0

        # 21 PSH301, DPIT301, DPSH301 > threshold => P301 OFF NOTE PSH301 not found

        # 22 LIT401 High => P301 or P302 OFF
        alert |= _inv(lit401h, p301off or p302off) # 0,0

        # 23 LIT401 Low => P301 or P302 ON
        alert |= _inv(lit401l, p301on or p302on) # 1,0

        # 24 LIT401 <= Low Low => P401 or P402 OFF => UV401 OFF ???

        # 25 P401 or P402 ON => FIT401 > delta
        alert |= _inv(p401on or p402on, s["FIT401"] > 0.1) # 8,1

        # 26 FIT401 Low Low => UV401 OFF
        alert |= _inv(fit401ll, uv401off) # 4,1

        # 27 P401 OFF => UV401 OFF
        alert |= _inv(p401off, not uv401off) # NOTE Rule inverted? => 5,0

        # 28 FIT401 Low Low => P403 or P404 OFF ( After some time 10 seconds)
        alert |= _inv(fit401ll, p403off or p404off) # 0,0

        # 29 AIT402 Low => P403 or P404 OFF
        alert |= _inv(ait402l, p403off or p404off) # 0,0

        # 30 AIT402 High => P403 or P404 ON AND LS401 NOT LL => P403 ON NOTE LS401 not found

        # 31 P401 ON => P501 ON AND UV401 ON => P501 ON ???

        # 32 P401 OFF => P501 OFF
        alert |= _inv(p401off, not p501off) # NOTE Rule inverted? => 5,0

        # 33 UV401 OFF => P501 OFF
        alert |= _inv(uv401off, p501off) # 4,0

        # 34 UV401 ON => P501 ON
        alert |= _inv(uv401on, p501on) # 0,0

        # 35 FIT401 Low Low => P501 OFF (After some time)
        alert |= _inv(fit401ll, p501off) # 4,1

        # 36 AIT504 NOT HIGH => MV501 OPEN NOTE MV501 not found

        # 37 LIT101 High High => P601 OFF
        alert |= _inv(lit101hh, p601off) # 0,0

        # 38 AIT202 < 7 => P601 OFF AND LS601 LOW => P601 OFF NOTE LS601 not found
        # 39 AIT202 > 7 => P601 ON AND LS601 NOT LOW => P601 ON NOTE LS601 not found


        ### 2 Distributed Detection

        # v2 =“LL”→ v3 = 1;for MV101
        alert |= _inv(lit101ll, mv101open) # 2,0

        # (v5 =‘‘H” or v8 =0 or v14 ≤δ)→ v4 =0; for P101 # NOTE works without FIT201
        #alert |= _inv(lit301h or mv201closed or s["FIT201"] <= 0.1, p101off) # 11,117

        # v2 = L → (v3 =1 AND v1 > δ)
        alert |= _inv(lit101l, mv101open and s["FIT101"] > 0.1) # 2,0

        # v2 = H → (v3 =0 AND v1 ≤ δ)
        alert |= _inv(lit101h, mv101closed and s["FIT101"] <= 1.0) # 4,3

        # v2 = “LL” → v4 = 0
        #alert |= _inv(lit101ll, p101off) # 8,61

        # v5 = “H” → v4 = 0
        alert |= _inv(lit301h, p101off) # 1,0

        # v4 = 1 → v8 = 1
        alert |= _inv(p101on, mv201open) # 0,0

        # v4 = 0 → v8 = 0
        #alert |= _inv(p101off, mv201closed) # 24,229

        # v5 = “HH” → v8 = 0
        alert |= _inv(lit301hh, mv201closed) # 3,0

        # v5 = “H” → v8 = 1
        #alert |= _inv(lit301h, mv201open) # 81,991

        # v5 = “H” → v8 = 1 AND v4 = 1
        #alert |= _inv(lit301h, mv201open and p101on) # 81,991

        # v5 = “HH” → v8 =0 AND v4 =0
        alert |= _inv(lit301hh, mv201closed and p101off) # 3,0

        # v5 = “LL” → v10 = 1
        alert |= _inv(lit301ll, p301on) # 6,0

        # v10 = 0 → v6 < δ
        #alert |= _inv(p301off, s["FIT301"] < 0.1) # 55,297

        # v6 > σ → v10 = 0
        alert |= _inv(s["FIT301"] > 0.1, p301off) # 0,0

        # v11 = “L” → v10 = 1 OR v9 = 1
        alert |= _inv(lit401ll, p301on or mv302open) # 2,0

        # v11 = “H” → v10 = 0
        alert |= _inv(fit401ll, p301off) # 0,0


        ### 3 HASE_Final_Paper

        #SD1 MV101 is Open,FIT101 >delta
        alert |= _inv(mv101open, s["FIT101"] > 0.1) # 0,0

        #SD2 LIT101 is Low,MV101 is Open
        alert |= _inv(lit101l, mv101open) # 2,0

        #SD3 LIT101 is High,MV101 is Close
        alert |= _inv(lit101h, mv101closed) # 2,0

        #SD4 LIT101 is LL (LowLow) P101 or P102 are Off
        alert |= _inv(lit101ll, p101off or p102off) # 3,0

        #SD5 LIT301 is Low,P101 or P102 is On
        alert |= _inv(lit301l, p101on or p102on) # 4,0

        #SD6 LIT301 is High,P101 or P102 is Off
        alert |= _inv(lit301h, p101off or p102off) # 0,0

        ### 4 A mechanistic fault detection

        # High Level: zk(LIT101) > 0.8 → iok(FIT101) = 0
        #alert |= _inv(lit101h, s["FIT101"] == 0) # 8,53

        # Low Level: zk(LIT101) < 0.5 → iok(FIT101) > 0
        alert |= _inv(lit101l, s["FIT101"] > 0) # 2,0

        # LowLow Level: zk(LIT101) < 0.25 → iok(FIT201) = 0
        #alert |= _inv(lit101ll, s["FIT201"] == 0) # 9,63


        ### 5 Generating invariants using
        # Tab. 6 & Tab. 7 were ignored since these invariants were "generated"

        # (4) True ⇒ LIT 101(k) < HH
        #alert |= _inv(True, not lit101hh) # 12,137

        # (5) LIT 101(k) > H ⇒ MV 101= CLOSED
        alert |= _inv(lit101h, mv101closed) # 2,0

        # (6) LIT 301(k) < L ⇒ P101= ON
        alert |= _inv(lit301l, p101on) # 4,0

        ### 6 A Systematic Framework

        # LIT101-H ⇒ MV101=OPEN
        #alert |= _inv(lit101h, mv101open) # 13,137

        # LIT301-L, LIT101-H, MV201=OPEN ⇒ P101=ON
        alert |= _inv(lit301l and lit101h and mv201open, p101on) # 0,0

        return alert, 0

    def load_trained_model(self):
        return True
