from shapes import Rect, Polygon

class Design(object):

    def __init__(self, layout_file):
        self.m_layout_file = layout_file

        self.m_rects = []
        self.m_polygons = []
        self.m_mask_rects = []
        self.m_num_true_rects = 0
        self.m_scale_numerator = 0
        self.m_scale_denominator = 0
        self.m_unit = ""
        self.m_cell_name = ""
        self.m_layer = ""
        self.m_min_width = 0

        self.parse()
        self.polygon2rect()

    @property
    def rects(self):
        return self.m_rects

    @property
    def polygons(self):
        return self.m_polygons

    @property
    def mask_rects(self):
        return self.m_mask_rects

    @property
    def num_true_rects(self):
        return self.m_num_true_rects

    @property
    def layout_file(self):
        return self.m_layout_file

    def add_rect(self, rect):
        self.m_rects.append(rect)

    def add_polygon(self):
        self.m_polygons.append(Polygon())

    def parse(self):
        file = self.file_content()
        # print(len(file))
        index = 0
        while index < len(file):
            token = file[index]
            if token == "BEGIN":
                index += 1
                continue
            elif token == "EQUIV":
                self.m_scale_numerator = int(file[index+1])
                self.m_scale_denominator = int(file[index + 2])
                self.m_unit = file[index+3]
                dummy = file[index+4]
                index += 5
            elif token == "CNAME":
                self.m_cell_name = file[index+1]
                index += 2

            elif token == "LEVEL":
                self.m_layer = file[index+1]
                index += 2
            elif token == "CELL":
                dummy = file[index+1]
                dummy = file[index+2]
                i = 0
                key_index = index + 3
                while i < (len(file) - key_index):
                    key_token = file[i + key_index]
                    if key_token == "RECT":
                        orient = file[i+1+key_index]
                        layer = file[i+2+key_index]
                        llx = int(file[i+3+key_index])
                        lly = int(file[i+4+key_index])
                        width = int(file[i+5+key_index])
                        height = int(file[i+6+key_index])
                        i += 7
                        index += 7
                        self.add_rect(Rect(llx, lly, llx+width, lly+height))
                    elif key_token == "PGON":
                        orient = file[i+1+key_index]
                        layer = file[i+2+key_index]
                        i += 3
                        index += 3
                        self.add_polygon()
                        while True:
                            head = file[i+key_index]
                            if head is not None:
                                if head[0].isdigit():
                                    point_x = int(file[i+key_index])
                                    point_y = int(file[i+1+key_index])
                                    self.m_polygons[-1].add_point(point_x, point_y)
                                    i += 2
                                    index += 2
                                else:
                                    break
                            else:
                                break
                    elif key_token == "ENDMSG":
                        i += 1
                        index += 1
                        break
            else:
                index += 1

    def file_content(self):

        with open(self.m_layout_file, 'r') as f:
            contents = f.readlines()

        process_contents = []
        for i in range(len(contents)):
            if contents[i][-1] == "\n":
                contents[i] = contents[i][:-1]

        for content in contents:
            content_list = content.split(" ")
            temp = [x.strip() for x in content_list if x.strip() != '']
            if temp is not None:
                process_contents += temp

        return process_contents

    def polygon2rect(self):
        self.m_num_true_rects = len(self.m_rects)
        for i in range(len(self.m_polygons)):
            rects = self.m_polygons[i].convert_rect()
            for j in range(len(rects)):
                self.add_rect(rects[j])

    def test_parser(self):
        for rect in range(len(self.m_rects)):
            self.m_mask_rects.append(rect)

    def write_glp(self, output_file):
        with open(output_file, 'w') as f:
            f.write("BEGIN\n")
            f.write("EQUIV {} {} {} +X,+Y\n".format(self.m_scale_numerator, self.m_scale_denominator, self.m_unit))
            f.write("CNAME {}\n".format(self.m_cell_name))
            f.write("LEVEL M1OPC\n\n")
            for mask_rect in self.m_mask_rects:
                f.write("   RECT N M1OPC ")
                f.write(str(mask_rect))
            f.write("ENDMSG\n")


if __name__ == "__main__":

    test_design = Design("../benchmarks/M1_test1.glp")
