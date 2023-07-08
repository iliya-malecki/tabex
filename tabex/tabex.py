import pandas as pd
import numpy as np
import layoutparser as lp
import pytesseract
import cv2
import matplotlib.pyplot as plt
from matplotlib import patches
import docx
from sklearn.cluster import AgglomerativeClustering

class Pipeline:
    def __init__(self, img_path):
        self.img_path = img_path
        self.img_color = None
        self.img_grayscale = None
        self.img_bin = None
        self.img_metadata: pd.DataFrame = None
        self.mutable = False


    def __getattribute__(self, __name: str) -> Any:
        if (
            not callable(super().__getattribute__(__name))
            or super().__getattribute__('mutable')
        ):
            return super().__getattribute__(__name)
        else:
            return object.__getattribute__(
                super().__getattribute__('copy')(),
                __name
            )


    def into_mutable(self):
        if self.mutable:
            raise ValueError('already mutable')
        c = self.copy()
        c.mutable = True
        return c


    def into_immutable(self):
        if not self.mutable:
            raise ValueError('already immutable')
        c = self.copy()
        c.mutable = False
        return c


    def copy(self):
        c = Pipeline(self.img_path)
        c.mutable = self.mutable
        if self.img_color is not None:
            c.img_color = self.img_color.copy()
            c.img_grayscale = self.img_grayscale.copy()
            c.img_bin = self.img_bin.copy()
            c.min_char_height = self.min_char_height
        if self.img_metadata is not None:
            c.img_metadata = self.img_metadata.copy()
        return c


    def load_image(self):
        self.img_color:np.ndarray = cv2.imread(self.img_path)
        self.img_grayscale:np.ndarray = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2GRAY)
        self.img_bin:np.ndarray = cv2.adaptiveThreshold(self.img_grayscale, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,15,5)
        self.min_char_height = get_min_char_height(self.img_path)
        return self


    def find_table_cells(
        self,
        min_cell_perimeter:float=None,
        bg_expected_col:'tuple[int,int,int]|None'=(250,250,250),
        bg_colorspace_chunk:'tuple[int,int,int]'=(10,10,10),
        bg_colorspace_tol=1,
        allowable_edge_distance=1,
        rectangle_angle_precision=1,
        polygon_precision=0.01,
        max_frame_fraction=0.8,
        ):
        '''
        Find all the contours in a binarized image and filter them based on
        some sanity checks, resulting in a dataframe of metadata about the cells.
        Checks include:
        - being a rectangle
        - not being smaller than the smallest letter
        - not being crooked
        - having an expected background color
        '''
        if min_cell_perimeter is None:
            min_cell_perimeter = self.min_char_height*4

        contours, hierarchy = cv2.findContours(self.img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        filtered_contours = []
        filtered_hierarchy = []
        for idx, (contour, relations) in enumerate(zip(contours, hierarchy[0])):

            path_len = cv2.arcLength(contour,True)
            if path_len < min_cell_perimeter or path_len > sum(self.img_bin.shape * 2) * max_frame_fraction:
                continue

            approx = cv2.approxPolyDP(contour, polygon_precision*path_len,True)
            if len(approx) != 4:
                continue

            if not check_path(approx.reshape(-1,2), tol=rectangle_angle_precision):
                continue

            x,y,w,h = cv2.boundingRect(approx)
            d = allowable_edge_distance
            if any([
                x <= d,
                y <= d,
                x+w >= self.img_bin.shape[1]-d,
                y+h >= self.img_bin.shape[0]-d
            ]):
                continue

            if (
                bg_expected_col is not None
                and not check_bg(
                    self.img_color[y:y+h, x:x+w],
                    bg_expected_col,
                    bg_colorspace_chunk,
                    bg_colorspace_tol
                )
            ):
                continue

            filtered_contours.append((x,y,w,h))
            filtered_hierarchy.append([idx, relations[-1]])

        items = (
            pd.DataFrame(
                filtered_hierarchy,
                columns=['index','table_idx']
            )
            .join(
                pd.DataFrame(
                    filtered_contours,
                    columns=['x','y','w','h'])
            )
            .set_index('index')
        )
        self.img_metadata = (
            items
            .drop(index=items['table_idx'],errors='ignore')
            .reset_index(drop=True)
        )
        return self


    def extract_tables(self):
        self.img_metadata: pd.DataFrame = (
            self.img_metadata
            .groupby('table_idx', group_keys=False)
            .apply(tableify, self.min_char_height)
            .sort_values(['table_y','table_x'])
        )
        return self


    def ocr(self, ocr_agent:str='tesseract', **ocr_kwargs):
        '''
        perform ocr on individual cells, focusing on recognition
        '''
        if ocr_agent == 'tesseract':
            ocr_agent = lp.TesseractAgent(**ocr_kwargs)

        self.img_metadata['text'] = (
            self.img_metadata
            [['x','y','h','w']]
            .astype(int)
            .apply(
                lambda row:
                ocr_agent.detect(
                    self.img_color[
                        row['y']:row['y']+row['h'],
                        row['x']:row['x']+row['w']
                    ]
                ).rstrip(),
                axis=1
            )
            .replace('',np.nan)
        )
        return self


    def cleanup(self):
        self.img_metadata = self.img_metadata[
            self.img_metadata
            .groupby('table_idx')['text']
            .transform(lambda s: s.isna().any())
        ]
        return self


    def to_excel(self, filename:str, merge_in_first_n=None):
        '''
        uses the metadata to build the replica of the table in question in excel,
        merging cells and setting sizes accordingly
        '''
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:

            for tablename, table in self.img_metadata.groupby('table_idx'):

                if table[['col','row']].isna().any().any():
                    raise ValueError(
                        'a table has nans for col/row:',
                        table[['col','row']]
                        .pipe(
                            lambda df:
                            df[df[['col','row']].isna().any(axis=1)]
                        )
                    )

                unstacked = to_df(table)

                unstacked.to_excel(
                    writer,
                    index=False,
                    header=False,
                    startcol=unstacked.columns.min(),
                    sheet_name=str(tablename)
                )
                sheet = writer.sheets[str(tablename)]

                for idx, cell in (
                    table
                    .head(merge_in_first_n)
                    .pipe(
                        lambda table:
                        table[
                            (table[['col_to_merge','row_to_merge']]>0).any(axis=1)]
                        .iterrows())
                ):
                    sheet.merge_range(
                        int(cell['row']),
                        int(cell['col']),
                        int(cell['row'] + cell['row_to_merge']),
                        int(cell['col'] + cell['col_to_merge']),
                        data=cell.fillna('')['text'])


                for colidx, width in table.groupby('col')['colwidths'].first().items():
                    sheet.set_column_pixels(first_col=colidx, last_col=colidx, width=width)

                for rowidx, height in table.groupby('row')['rowheights'].first().items():
                    sheet.set_row_pixels(rowidx, height)


    def to_docx(self, filename:str, merge_in_first_n=None):
        '''
        uses the metadata to build the replica of the table in question in docx,
        merging cells and setting sizes accordingly
        '''

        if merge_in_first_n is None:
            merge_in_first_n = np.nan

        document = docx.Document()

        for tablename, table in self.img_metadata.groupby('table_idx'):

            if table[['col','row']].isna().any().any():
                raise ValueError(
                    'a table has nans for col/row:',
                    table[['col','row']]
                    .pipe(
                        lambda df:
                        df[df[['col','row']].isna().any(axis=1)]
                    )
                )

            doctable = document.add_table(
                rows=table['row'].max()+1,
                cols=table['col'].max()+1
            )
            doctable.style = 'TableGrid'
            doctable.autofit = False
            doctable.alignment = docx.enum.table.WD_TABLE_ALIGNMENT.CENTER

            for colidx, width in table.groupby('col')['colwidths'].first().items():
                for c in doctable.columns[colidx].cells:
                    c.width = docx.shared.Pt(width)

            for rowidx, height in table.groupby('row')['rowheights'].first().items():
                for c in doctable.rows[rowidx].cells:
                    c.height = docx.shared.Pt(height)


            for idx, cell in table.fillna({'text':''}).iterrows():

                if (
                    (cell[['col_to_merge','row_to_merge']] > 0).any()
                    and not cell['row']>=merge_in_first_n
                ):

                    merged = (
                        doctable
                        .cell(cell['row'], cell['col'])
                        .merge(
                            doctable.cell(
                                cell['row']+cell['row_to_merge'],
                                cell['col']+cell['col_to_merge']
                            )
                        )
                    )
                    merged.text = str(cell['text'])

                else:
                    doctable.cell(cell['row'], cell['col']).text = str(cell['text'])


        document.save(filename)


def check_path(path:np.ndarray, tol=1) -> None:
    '''
    path: array of shape (points, dimensions of each point)
    tol: tolerance in degrees, describes how much the path can deviate from a rectangle
    '''
    def angle(side1, side2):
        side1 = side1/np.linalg.norm(side1)
        side2 = side2/np.linalg.norm(side2)
        return np.degrees(np.arccos(np.dot(side1, side2)))

    lens = path - np.roll(path,1, axis=0)
    angles_ok = [
        (90 - tol) < angle(side1,side2) < (90 + tol)
        for side1, side2
        in zip(lens, np.roll(lens,1, axis=0))
    ]
    return all(angles_ok)


def show(
    df:pd.DataFrame,
    img:np.ndarray,
    annotate=True,
    figsize=(20,20),
    color='red',
    annotation_color='darkred',
    linewidth=1
):
    '''
    Display the detected cells dataframe as a set of rectangles on the image
    '''

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, 'gray')
    for idx, (x, y, w, h) in zip(df.index, df[['x', 'y', 'w', 'h']].values):
        ax.add_patch(
            patches.Rectangle(
                (x-1,y-1),
                w, h,
                linewidth=linewidth,
                edgecolor=color,
                facecolor='none'
            )
        )
        if annotate:
            plt.annotate(
                idx,
                (x,y),
                xytext=(0,-10),
                textcoords='offset pixels',
                color=annotation_color
            )


def check_bg(
    img:np.ndarray,
    color,
    granularity,
    dist=1
):
    '''
    Split the image into bins by color and check that the
    distance from the expected color to the most popular color is
    less than the specified number of bins
    '''

    def to_rgb(input):
        if np.isscalar(input):
            return cv2.cvtColor(
                np.array(input, dtype=np.uint8).reshape(1,1),
                cv2.COLOR_GRAY2RGB
            ).ravel()
        else:
            return np.array(input, dtype=np.uint8)

    def to_gray(input):
        if np.isscalar(input):
            return np.array(input, dtype=np.uint8)
        else:
            return cv2.cvtColor(
                np.array(input, dtype=np.uint8).reshape(1,1,3),
                cv2.COLOR_RGB2GRAY
            ).ravel()


    if len(img.shape) == 3:
        img = img.reshape(-1, 3)
        color = to_rgb(color)
        granularity = to_rgb(granularity)

    elif len(img.shape) == 2:
        img = img.reshape(-1, 1)
        color = to_gray(color)
        granularity = to_gray(granularity)

    hist, edges = np.histogramdd(img, bins=(255/granularity).astype(int))
    idxmax = np.unravel_index(np.argmax(hist), hist.shape)

    target_idx = np.array([
        np.abs(t - ar).argmin()
        for t, ar in zip(color, edges)
    ])

    return np.all(np.abs(idxmax - target_idx) <= dist)


def get_min_char_height(file:str) -> float:
    return (
        pytesseract.image_to_data(file, output_type=pytesseract.Output.DATAFRAME)
        .replace(r'\s+',np.nan, regex=True)
        .replace(r'',np.nan)
        .dropna(subset=['text'])
        ['height']
        .quantile(0.01)
    )


def hclust(s:pd.DataFrame, threshold:float):

    if isinstance(s, pd.Series):
        s = s.to_frame()

    pred = (
        pd.Series(
            AgglomerativeClustering(
                distance_threshold=threshold,
                n_clusters=None
            )
            .fit_predict(s),
            name='pred',
            index=s.index
        )
        .sort_values(key=lambda pred: s.squeeze())
    )
    # relying on the fact that sorted input will have clusters grouped together
    changepoints = pred != pred.shift(1)
    assert changepoints.sum() == pred.max()+1

    return changepoints.cumsum() - 1


def tableify(cells:pd.DataFrame, min_char_height:float):
    '''
        Calculate which rows and columns the cells belong to,
        and how to merge them
    '''

    if len(cells) > 1:
        cells['col'] = hclust(cells['x'], min_char_height)
        cells['row'] = hclust(cells['y'], min_char_height)
    else:
        cells['col'] = 0
        cells['row'] = 0

    colticks = cells.groupby('col')['x'].mean()
    rowticks = cells.groupby('row')['y'].mean()

    colwidths = colticks.diff().shift(-1).fillna(cells.query('col==col.max()')['w'].mean())
    rowheights = rowticks.diff().shift(-1).fillna(cells.query('row==row.max()')['h'].mean())


    cells['col_to_merge'] = (
        (cells['x'] + cells['w'])
        .apply(
            lambda col_end:
            ((colticks + colwidths - col_end).abs() < min_char_height)
            .pipe(lambda s: s[s].index.max())
        )
        .pipe(lambda x: x - cells['col'])
        .fillna(0)
        .astype(int)
    )

    cells['row_to_merge'] = (
        (cells['y'] + cells['h'])
        .apply(
            lambda row_end:
            ((rowticks + rowheights - row_end).abs() < min_char_height)
            .pipe(lambda s: s[s].index.max())
        )
        .pipe(lambda x: x - cells['row'])
        .fillna(0)
        .astype(int)
    )

    cells['colticks'] = colticks.reindex(cells['col']).values
    cells['rowticks'] = rowticks.reindex(cells['row']).values

    cells['colwidths'] = colwidths.reindex(cells['col']).values
    cells['rowheights'] = rowheights.reindex(cells['row']).values

    cells['table_y'] = cells['y'].min()
    cells['table_x'] = cells['x'].min()
    cells['table_w'] = cells['w'].sum()
    cells['table_h'] = cells['h'].sum()

    return cells



def to_df(metadata:pd.DataFrame):
    '''
    builds the table as it looks in the source (or close to it)
    using the column and row metadata
    '''
    if 'text' not in metadata:
        raise ValueError(
            'This dataframe lacks the column "text", which means no ocr was performed'
        )
    return (
        metadata
        .set_index(['row','col'])
        .sort_index()
        ['text']
        .unstack()
    )



if __name__ == '__main__':
    (
        Pipeline('./example.png')
        .load_image()
        .find_table_cells()
        .extract_tables()
        .ocr(config='--psm 6')
        .to_excel('example.xlsx')
    )
