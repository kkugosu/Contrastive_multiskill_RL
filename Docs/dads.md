## DADS

this paper learns skills from "current state and next state pair"

$$1. \ sample \ skill \ from \ p(z)$$

$$2. \ simulate \ in \ environment \ while \ memory \ become \ full$$

$$3. \ compute \ q_\phi(s' | s, z) - q_\phi(s' | s)\ from \ discriminator \ q_\phi $$

$$4. \ improve \ value \ q_\phi(s' | s, z) - q_\phi(s' | s)\ by \ update \ \phi$$

$$5. \ set \ reward \ as \ q_\phi(s' | s, z)- q_\phi(s' | s) $$

$$6. \ update \ sac \ with \ policy \ z \ to \ maximize \ reward$$


