# app/__init__.py
# -*- coding: utf-8 -*-

import os
import json
from app.src.utils.utils import DocumentDB, IBMCOS

# definición de constantes a usar en la app
client = None
cos = None
COS_ENDPOINT = "https://s3.ap.cloud-object-storage.appdomain.cloud"
COS_AUTH_ENDPOINT = "https://iam.cloud.ibm.com/identity/token"

# ruta del directorio ráiz del proyecto
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# conexión a servicios de IBM Cloud (VCAP_SERVICES) usando variables de entorno o fichero local
# Variable de entorno (Despliegue)
if 'VCAP_SERVICES' in os.environ:
    # carga de la variable de entorno VCAP_SERVICES
    vcap = json.loads(os.getenv('VCAP_SERVICES'))
    # se busca el servicio de IBM Cloudant (debe estar conectado en IBM Cloud a nuestra app)
    if 'cloudantNoSQLDB' in vcap:
        creds = vcap['cloudantNoSQLDB'][0]['credentials']
        api_key = creds['apikey']
        host = creds['host']
        url = creds['url']
        username = creds['username']
        # se crea la conexión a IBM Cloudant
        client = DocumentDB(username, api_key)

    # se busca el servicio de IBM COS (debe estar conectado en IBM Cloud a nuestra app)
    if 'cloud-object-storage' in vcap:
            creds = vcap['cloud-object-storage'][0]['credentials']
            endpoint_url = COS_ENDPOINT
            ibm_service_instance_id = creds['resource_instance_id']
            ibm_api_key_id = creds['apikey']
            # se crea la conexión a IBM COS
            cos = IBMCOS(ibm_api_key_id, ibm_service_instance_id, COS_AUTH_ENDPOINT, endpoint_url)

# Variable de entorno (Local)
elif os.path.isfile('vcap-local.json'):
    with open('vcap-local.json') as f:
        vcap = json.load(f)
        if 'cloud-object-storage' in vcap['services']:
            creds = vcap['services']['cloud-object-storage'][0]['credentials']
            endpoint_url = creds['endpoints']
            ibm_service_instance_id = creds['resource_instance_id']
            ibm_api_key_id = creds['apikey']
            # Constantes correspondientes a valores de IBM COS

            cos = IBMCOS(ibm_api_key_id, ibm_service_instance_id, COS_AUTH_ENDPOINT, endpoint_url)

        if 'cloudantNoSQLDB' in vcap['services']:
            creds = vcap['services']['cloudantNoSQLDB'][0]['credentials']
            api_key = creds['apikey']
            host = creds['host']
            url = creds['url']
            username = creds['username']
            # se crea la conexión a IBM Cloudant
            client = DocumentDB(username, api_key)
