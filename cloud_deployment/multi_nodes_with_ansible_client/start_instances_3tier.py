# http://docs.openstack.org/developer/python-novaclient/ref/v2/servers.html
import time, os, sys, random, re
import inspect
from os import environ as env

from  novaclient import client
import keystoneclient.v3.client as ksclient
from keystoneauth1 import loading
from keystoneauth1 import session


flavor = "ssc.small" 
private_net = "SNIC 2020/20-42 Internal IPv4 Network"
floating_ip_pool_name = None
floating_ip = None
image_name = "cb36ca0f-9be4-4216-b714-7582ef726ead"

identifier = random.randint(1000,9999)

loader = loading.get_plugin_loader('password')

auth = loader.load_from_options(auth_url=env['OS_AUTH_URL'],
                                username=env['OS_USERNAME'],
                                password=env['OS_PASSWORD'],
                                project_name=env['OS_PROJECT_NAME'],
                                project_domain_name=env['OS_USER_DOMAIN_NAME'],
                                #project_id=env['OS_PROJECT_ID'],
                                user_domain_name=env['OS_USER_DOMAIN_NAME'])

sess = session.Session(auth=auth)
nova = client.Client('2.1', session=sess)
print ("user authorization completed.")

image = nova.glance.find_image(image_name)

flavor = nova.flavors.find(name=flavor)

if private_net != None:
    net = nova.neutron.find_network(private_net)
    nics = [{'net-id': net.id}]
else:
    sys.exit("private-net not defined.")

#print("Path at terminal when executing this file")
#print(os.getcwd() + "\n")
cfg_file_path =  os.getcwd()+'/fast-cloud-cfg.txt'
if os.path.isfile(cfg_file_path):
    userdata_fast = open(cfg_file_path)
else:
    sys.exit("fast-cloud-cfg.txt is not in current working directory")

cfg_file_path =  os.getcwd()+'/medium-cloud-cfg.txt'
if os.path.isfile(cfg_file_path):
    userdata_medium = open(cfg_file_path)
else:
    sys.exit("medium-cloud-cfg.txt is not in current working directory")

cfg_file_path =  os.getcwd()+'/slow-cloud-cfg.txt'
if os.path.isfile(cfg_file_path):
    userdata_slow = open(cfg_file_path)
else:
    sys.exit("slow-cloud-cfg.txt is not in current working directory")    

secgroups = ['default']

print ("Creating instances ... ")
instance_fast = nova.servers.create(name="TZ_fast_tier", image=image, flavor=flavor, key_name='TianruZ',userdata=userdata_fast, nics=nics,security_groups=secgroups)
instance_medium = nova.servers.create(name="TZ_medium_tier", image=image, flavor=flavor, key_name='TianruZ',userdata=userdata_medium, nics=nics,security_groups=secgroups)
instance_slow = nova.servers.create(name="TZ_slow_tier", image=image, flavor=flavor, key_name='TianruZ',userdata=userdata_slow, nics=nics,security_groups=secgroups)
inst_status_fast = instance_fast.status
inst_status_medium = instance_medium.status
inst_status_slow = instance_slow.status

print ("waiting for 10 seconds.. ")
time.sleep(10)

while inst_status_fast == 'BUILD' or inst_status_slow == 'BUILD':
    print ("Instance: "+instance_fast.name+" is in "+inst_status_fast+" state, sleeping for 5 seconds more...")
    print ("Instance: "+instance_medium.name+" is in "+inst_status_medium+" state, sleeping for 5 seconds more...")
    print ("Instance: "+instance_slow.name+" is in "+inst_status_slow+" state, sleeping for 5 seconds more...")
    time.sleep(5)
    instance_fast = nova.servers.get(instance_fast.id)
    inst_status_fast = instance_fast.status
    instance_medium = nova.servers.get(instance_medium.id)
    inst_status_medium = instance_medium.status
    instance_slow = nova.servers.get(instance_slow.id)
    inst_status_slow = instance_slow.status

ip_address_fast = None
for network in instance_fast.networks[private_net]:
    if re.match('\d+\.\d+\.\d+\.\d+', network):
        ip_address_fast = network
        break
if ip_address_fast is None:
    raise RuntimeError('No IP address assigned!')

ip_address_medium = None
for network in instance_medium.networks[private_net]:
    if re.match('\d+\.\d+\.\d+\.\d+', network):
        ip_address_medium = network
        break
if ip_address_medium is None:
    raise RuntimeError('No IP address assigned!')

ip_address_slow = None
for network in instance_slow.networks[private_net]:
    if re.match('\d+\.\d+\.\d+\.\d+', network):
        ip_address_slow = network
        break
if ip_address_slow is None:
    raise RuntimeError('No IP address assigned!')

print ("Instance: "+ instance_fast.name +" is in " + inst_status_fast + " state" + " ip address: "+ ip_address_fast)
print ("Instance: "+ instance_medium.name +" is in " + inst_status_medium + " state" + " ip address: "+ ip_address_medium)
print ("Instance: "+ instance_slow.name +" is in " + inst_status_slow + " state" + " ip address: "+ ip_address_slow)
