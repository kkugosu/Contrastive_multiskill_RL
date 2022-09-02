## VIC

this paper learns skills from "initial state and last state pair"

$$1. \ sample \ skill \ from \ p(z)$$

$$2. \ simulate \ in \ environment \ while \ memory \ become \ full$$

$$3. \ compute \ q_\phi(z | s_0, s_t)\ and \ q_\phi(z | s_0, s_0) \ from \ discriminator \ q_\phi$$

$$4. \ improve \ value \ q_\phi(z | s_0, s_t)\ - \ q_\phi(z | s_0, s_0) \ by \ update \ \phi$$

$$5. \ set \ reward \ as \ q_\phi(z | s_0, s_t)\ - \ q_\phi(z | s_0, s_0)$$

$$6. \ update \ sac \ with \ policy \ z \ to \ maximize \ reward$$
