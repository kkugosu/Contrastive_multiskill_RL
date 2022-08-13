import torch


class Train:
    def __init__(self, env, writer, train_iter, *model):
        self.t_i = train_iter
        self.model = model
        self.buffer = buffer.Simulate(self.env, self.policy, step_size=self.e_trace, done_penalty=self.d_p)

    def loading(self):
        print("loading")
        i = 0
        while i < len(self.model):
            self.model[i].load_state_dict(torch.load(self.PARAM_PATH + '/' + str(i) + '.pth'))
            i = i + 1
        print("loading complete")

    def saving(self):
        i = 0
        while i < len(self.model):
            torch.save(self.model[i].state_dict(), self.PARAM_PATH + '/' + str(i) + '.pth')
            i = i + 1

    def train(self):
        self.loading()
        i = 0
        while i < self.t_i:
            print(i)
            i = i + 1
            self.buffer.renewal_memory(self.ca, self.data, self.dataloader)
            loss = self.update()

            print("loss = ", loss)
            j = 0
            while j < len(loss):
                self.writer.add_scalar("loss " + str(j), loss, i)
                j = j + 1
            self.writer.add_scalar("performance", self.buffer.get_performance(), i)
            self.saving()

        i = 0
        while i < len(self.model):
            for param in self.model[i].parameters():
                print("-----------" + str(i) + "-------------")
                print(param)
            i = i + 1

        self.env.close()
        self.writer.flush()
        self.writer.close()