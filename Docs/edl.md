## EDL

this paper match skill to state by vae
and learn policy with reward given by that skill

$$1. \ get \ distribution \ of \ state \ p(s)$$

$$2. \ extract \ state \ and \ learn \ skill \ from \ that \ state \ by \ vae $$

$$3. \ simulate \ with \ skills \ in \ environment \ while \ memory \ become \ full$$

$$4. \ set \ reward \ as \ q_\phi(s | z), \ q_\phi \ is \ decoder $$

$$5. \ update \ sac \ with \ policy \ z \ to \ maximize \ reward$$


