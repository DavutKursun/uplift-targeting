---
title: uplift-targeting
emoji: 🎯
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
license: mit
pinned: false
short_description: Who should get the campaign — uplift, segment, treat/skip
---

# uplift-targeting — who should get the campaign? Causal effect, not response probability

Scores a customer with two models side by side: a response model, which ranks by who would visit, and an S-learner, which
ranks by the change the ad causes, both trained on the Criteo Uplift Prediction Dataset (a randomized experiment, 85% treated)
and tested once on 800,004 held-out customers. At a 10% budget the uplift model finds 63 incremental visits per 1,000 contacts
against 49 for the response model and 9 for random targeting, a paired difference of +14 [6.5, 21.6]; by 20% the two tie, and
under the stated economics every policy loses money past 14%. Segments: persuadable, sure thing, lost cause, and
no_detectable_effect (the textbook sleeping dog, renamed because held-out data did not confirm any harm).

The code is MIT-licensed; **the data is Criteo's, CC BY-NC-SA 4.0 (non-commercial)** — cite Diemert et al., AdKDD 2018. Features
are anonymous. Historical experiment, not a live marketing system; nothing you type is stored. Open **/ui** for the demo,
**/docs** for the API; repository: https://github.com/DavutKursun/uplift-targeting
