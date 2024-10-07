import fitz


def rotate_bbox(rect, angle, page_width, page_height):
    x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
    if angle == 90:
        new_x0 = y0
        new_y0 = page_width - x1
        new_x1 = y1
        new_y1 = page_width - x0
        return fitz.Rect(new_x0, new_y0, new_x1, new_y1)
    elif angle == 180:
        new_x0 = page_width - x1
        new_y0 = page_height - y1
        new_x1 = page_width - x0
        new_y1 = page_height - y0
        return fitz.Rect(new_x0, new_y0, new_x1, new_y1)
    elif angle == 270:
        new_x0 = page_height - y1
        new_y0 = x0
        new_x1 = page_height - y0
        new_y1 = x1
        return fitz.Rect(new_x0, new_y0, new_x1, new_y1)
    return rect


def get_textbox(page, rect, overlap_threshold=0.5):
    """
    subselect words from words list that subject to condition: intersect_area (rect,word area) > overlap_threshold * word area:
    The resulting sublist is then converted to a string by space between.
    Params:
    page: pymupdf page object
    The first 4 entries are the word's rectangle coordinates, the last 3 are just
    technical info (block number, line number, word number).
    The term 'word' here stands for any string without space.
    rect: area to get word type https://pymupdf.readthedocs.io/en/latest/rect.html
    overlap_threshold : Default = 0.5
    """
    words = get_sorted_words(page)

    # rotate_matrix = fitz.Matrix(fitz.Identity).prerotate(page.rotation)
    # rect = rect.transform(rotate_matrix)
    rect = rotate_bbox(rect, page.rotation, page.rect.width, page.rect.height)
    contained_words = [w[1] for w in words if
                       w[0].get_area() * overlap_threshold < fitz.Rect(w[0]).intersect(rect).get_area()]
    return " ".join(contained_words)


def get_sorted_words(page):
    """
    Sort words in a pymupdf page by the read order
    rotate coordinate first if the page.rotation != 0

    Params:
        page: pymupdf page object
    return
    words: ordered all words on page in a list of lists. Each word is represented by:
    [rect(x0, y0, x1, y1), word]
    """
    # extract words, sorted by bottom, then left coordinate
    rotate_matrix = fitz.Matrix(fitz.Identity).prerotate(page.rotation)
    words = [
        (fitz.Rect(w[:4]).transform(rotate_matrix), w[4]) for w in page.get_text("words")
    ]
    if not words:
        return words
    words = sorted(words, key=lambda x: (x[0][1], x[0][0]))

    lines = []  # list of reconstituted lines
    line = [words[0]]  # current line
    lrect = words[0][0]  # the line's rectangle

    # walk through the words
    for wr, text in words[1:]:
        # if this word matches top or bottom of the line, append it
        if abs(lrect.y0 - wr.y0) <= 3 or abs(lrect.y1 - wr.y1) <= 3:
            line.append((wr, text))
            lrect |= wr
        else:
            # output current line and re-initialize
            # note that we sort the words in current line first
            line = sorted(line, key=lambda w: w[0].x0)
            lines.append(line)
            line = [(wr, text)]
            lrect = wr

    # also append last unfinished line
    line = sorted(line, key=lambda w: w[0].x0)
    lines.append(line)

    # sort all lines vertically
    lines.sort(key=lambda l: (l[0][0].y1))

    # reverse to words list
    reversed_matrix = fitz.Matrix(fitz.Identity).prerotate(-page.rotation)
    words = [(w[0].transform(reversed_matrix), w[1]) for line in lines for w in line]
    return words