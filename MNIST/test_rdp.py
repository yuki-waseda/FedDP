

orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +list(range(5, 64)) + [128, 256, 512])

rdp = compute_rdp(float(mt/len(self.clients)), self.sigmat, i, self.orders)
_,delta_spent, opt_order = get_privacy_spent(self.orders, rdp, target_eps=self.epsilon)
