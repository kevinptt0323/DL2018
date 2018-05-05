
class Summary():
    def __init__(self):
        self.result = []

    def add(self, row, key, val):
        self.result.append((row, key, val))

    def write(self, filename):
        self.result.sort()
        last_row = self.result[0][0]-1

        keys = list(set([log[1] for log in self.result]))

        with open(filename, "w") as f:
            row_data = [""] + keys
            for row, key, val in self.result:
                if row != last_row:
                    f.write(','.join(map(str, row_data)) + '\n')
                    row_data = [""] * (len(keys)+1)
                    row_data[0] = row
                    row = last_row

                row_data[keys.index(key) + 1] = val
