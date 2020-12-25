class ApiPaths(object):

    _gdpPerCapita = '/gdp-per-capita'

    def get_gdpPerCapita(self):
        return type(self)._gdpPerCapita

    GdpPerCapita = property(get_gdpPerCapita)
