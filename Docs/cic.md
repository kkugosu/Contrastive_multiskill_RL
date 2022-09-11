
## CIC

we embed "current state and next state pair" and "skill" to abstract state and maximize entropy from "current state and next state pair" distribution 

$$1. \ sample \ skill \ from \ p(z)$$

$$2. \ simulate \ in \ environment \ while \ memory \ become \ full$$

$$3. \ learn \ "state \ pair \ and \ skill" \  embedding \ from \ key \ and \ query \ network$$

$$4. \ set \ reward \ as \  entropy(s_pair, s_pair') + log(p(pair|skill))$$

$$5. \ update \ sac \ with \ policy \ z \ to \ maximize \ reward$$
