## EDL

this paper match skill to state my vae
and learn policy by reward given by skill

$$1. \ sample \ skill \ from \ p(z)$$

$$2. \ step \ in \ environment \ while \ memory \ become \ full$$

$$3. \ extract \ state \ from \ memory \ to \ mapping \ state \ to \ skill  \ by \ vae $$

$$5. \ set \ reward \ as \ q_\phi(s | z), \ q_\phi \ is \ decoder $$

$$6. \ update \ sac \ with \ policy \ z \ to \ maximize \ reward$$


