import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import xml.etree.ElementTree as ET
import requests
#import PIL
#from PIL import Image
from astropy.io import fits
from astropy.wcs import WCS
import os
import gc
import urllib


def download_sloan_image(ra, dec, objid, imdir):
    

    if str(objid) + '.jpeg' in os.listdir(imdir):
                
        print('Already downloaded'   )

    else:
        
        url = 'http://skyservice.pha.jhu.edu/DR10/ImgCutout/getjpeg.aspx?ra=' 
        url += str(ra) 
        url += '&dec=' 
        url += str(dec) 
        url += '&scale=0.25&width=256&height=256&opt='
        
        print(url)
        
        file_name = imdir + str(objid) + '.jpeg'
        u = urllib.request.urlopen(url)
        f = open(file_name, 'wb')
        file_size = int(u.getheader("Content-Length"))
        print("Downloading: %s Bytes: %s" % (file_name, file_size))
        
        file_size_dl = 0
        block_sz = 8192
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break
        
            file_size_dl += len(buffer)
            f.write(buffer)
            status = r"%10d  [%3.2f%%]" % (file_size_dl
            , file_size_dl * 100. / file_size)
            status = status + chr(8)*(len(status)+1)
            print(status)
        
        f.close()
        
        gc.collect()

def plot_image_sdss(galid, imdir):
    image = mpimg.imread(imdir + str(galid) + '.jpeg', format = 'jpeg')
    plt.imshow(image)


def download_and_plot_image_sdss(ra, dec, galid, imdir='.', ax=plt.gca()):
    if str(galid) + '.jpeg' not in os.listdir(imdir):
        download_sloan_image(ra, dec, galid, imdir)
    
    image = mpimg.imread(imdir + str(galid) + '.jpeg', format = 'jpeg')
    ax.imshow(image,zorder = 10)

def download_and_plot_image_sdss_zoom(ra, dec, galid, imdir):
    if str(galid) + '.jpeg' not in os.listdir(imdir):
        download_sloan_image(ra, dec, galid, imdir)
    
    image = mpimg.imread(imdir + str(galid) + '.jpeg', format = 'jpeg')
    
    plt.imshow(image , zorder = 10)

        
def download_manga_image(plate, ifudsgn, objid, imdir):
    import urllib2

    if str(objid) + '.png' in os.listdir(imdir):
                
        print('Already downloaded')

    else:
        
        url = 'https://data.sdss.org/sas/dr14/manga/spectro/redux/v2_1_2/'
        url+= str(plate) + '/stack/images/' + str(ifudsgn) + '.png'
        
        print(url)
        
        file_name = imdir + str(objid) + '.png'
        u = urllib2.urlopen(url)
        f = open(file_name, 'wb')
        meta = u.info()
        file_size = int(meta.getheaders("Content-Length")[0])
        print("Downloading: %s Bytes: %s" % (file_name, file_size))
        
        file_size_dl = 0
        block_sz = 8192
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break
        
            file_size_dl += len(buffer)
            f.write(buffer)
            status = r"%10d  [%3.2f%%]" % (file_size_dl
            , file_size_dl * 100. / file_size)
            status = status + chr(8)*(len(status)+1)
            print(status)
        
        f.close()
        
        gc.collect()


#def plot_image_galex(ra,dec):
    
    #r = requests.request('GET', 'http://galex.stsci.edu/gxWS/SIAP/gxSIAP.aspx?POS=%s,%s&SIZE=0.1' % (ra, dec))
    #data = ET.XML(r.content)
    #resource = data.find('{http://www.ivoa.net/xml/VOTable/v1.1}RESOURCE')
    #table = resource.find('{http://www.ivoa.net/xml/VOTable/v1.1}TABLE').find(
        #'{http://www.ivoa.net/xml/VOTable/v1.1}DATA').find(
        #'{http://www.ivoa.net/xml/VOTable/v1.1}TABLEDATA')
    
    #img_dict = dict()
    #for child in table.getchildren():
        #img_dict[child[0].text+'_'+child[14].text] = child[20].text
    
    #if 'GII_NUV' in img_dict.keys():
        #survey = 'GII'
    #elif 'DIS_NUV' in img_dict.keys():
        #survey = 'DIS'
    #elif 'NGS_NUV' in img_dict.keys():
        #survey = 'NGS'
    #elif 'MIS_NUV' in img_dict.keys():
        #survey = 'MIS'
    #else: survey = 'AIS'
    
    #print img_dict.keys()    
    #print 'Chosen survey: %s' % survey
    
    #img = Image.open(StringIO(requests.request('GET', img_dict['%s_FUV+NUV' % survey]).content))
    #w = WCS(fits.getheader(img_dict["%s_NUV" % survey]))
    #pos = w.wcs_world2pix(float(ra), float(dec), 0)
    #img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM).crop(box = [pos[0] - 35, pos[1] - 35, pos[0] + 35, pos[1] + 35])    
    #plt.imshow(img.transpose(PIL.Image.FLIP_LEFT_RIGHT).rotate(-180),zorder=10)
    
#    del img, r, data, resource, table
       
       
def plot_galex_and_sdss_image(objid, ra, dec, flag, plotdir, imdir):
    for i in range(flag.sum()):
        print(i)
        if str(objid[flag][i]) + '.jpeg' in os.listdir(imdir):        
        
            p1 = plt.subplot(121)
            plot_image_sdss(objid[flag][i], imdir = imdir)
            plt.setp(p1.get_xticklabels(), visible = False)
            plt.setp(p1.get_yticklabels(), visible = False)
            
            p2 = plt.subplot(122)
            img_dict, img = plot_image_galex(ra[flag][i],dec[flag][i])
            plt.setp(p2.get_xticklabels(), visible = False)
            plt.setp(p2.get_yticklabels(), visible = False)
        
            plt.savefig(plotdir + 'galex+sdss_' + str(objid[flag][i]) + '.png')
            plt.show()
        

def plot_manga_image(imdir, galid):
    image = mpimg.imread(imdir + str(galid) + '.png', format = 'png')
    plt.imshow(image)
   
