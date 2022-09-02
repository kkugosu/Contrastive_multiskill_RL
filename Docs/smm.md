
## SMM

this paper learns skills from state and add penalty to high density state

$$1. \ sample \ skill \ from \ p(z)$$

$$2. \ simulate \ in \ environment \ while \ memory \ become \ full$$

$$3. \ compute \ q_\phi(z | s) \ from \ discriminator \ q_\phi $$

$$4. \ improve \ value \ q_\phi(z | s) \ by \ update \ \phi$$

$$5. \ set \ reward \ as \ q_\phi(z | s) - (p(s) - mean(p(s)))$$

$$6. \ update \ sac \ with \ policy \ z \ to \ maximize \ reward$$


