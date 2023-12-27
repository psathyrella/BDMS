class RandomizedSet:
    def __init__(self):
        self._name_to_idx = {}
        self._idx_to_name = {}
        self.size = 0

    def add(self, name):
        if name in self._name_to_idx:
            return False
        self._name_to_idx[name] = self.size
        self._idx_to_name[self.size] = name
        self.size += 1
        return True

    def remove(self, del_name):
        if del_name not in self._name_to_idx:
            return False
        # Swap the element with the last element
        last_name, del_idx = (
            self._idx_to_name[self.size - 1],
            self._name_to_idx[del_name],
        )
        self._name_to_idx[last_name], self._idx_to_name[del_idx] = del_idx, last_name
        # Remove the last element
        del self._name_to_idx[del_name]
        del self._idx_to_name[self.size - 1]
        self.size -= 1
        return True

    def choice(self, rng):
        random_idx = rng.choice(self.size)
        return self._idx_to_name[random_idx]

    def as_list(self):
        return list(self._name_to_idx.keys())
