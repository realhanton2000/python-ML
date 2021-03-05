import cherrypy
from number import predictX

class PredictX(object):
    @cherrypy.expose
    def X(self, index=100):
        return "".join(str(predictX(int(index))))

if __name__ == '__main__':
    cherrypy.quickstart(PredictX())