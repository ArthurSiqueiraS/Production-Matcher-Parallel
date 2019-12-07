#!/usr/bin/env python
# coding: utf-8

import xml.etree.ElementTree as ET
from modules.normalization import Normalizer
import os
import shutil
import torch
import torch.nn as nn
import modules.mlp as mlp
from fuzzywuzzy import fuzz
import sys
import zipfile
from multiprocessing import Pool
from functools import partial, reduce
import time 
import numpy as np
import pandas as pd
import re

def specific_fields(type):
    fields = {
        "field_type": '',
        "country": '',
        "homepage": ''
    } 
    
    if type in ['artigo_em_evento', 'artigo_em_periodico']:
        fields["kind"] = 'bibliografica'
        fields["homepage"] = "-DO-TRABALHO"
        
        if type == 'artigo_em_evento':
            fields["field_type"] = "-DO-TRABALHO"
            fields["country"] = "-DO-EVENTO"
        elif type == 'artigo_em_periodico':
            fields["field_type"] = "-DO-ARTIGO"
            fields["country"] = "-DE-PUBLICACAO"
    else: 
        fields["kind"] = 'orientacao'
        
    return fields.get('kind'), fields.get('field_type'), fields.get('country'), fields.get('homepage')

def normalize_author_name(name):
  name = name.lower().strip()
  while not re.search("[\w\.]$", name):
    name = name[:-1]

  if ',' in name:
    last_name = name.split(',')[0].split()[-1]
    first_name_letter = name.split(',')[1].split()[0][0]

  else:
    last_name = name.split()[-1]
    first_name_letter = name.split()[0][0]

  return f"{first_name_letter}{last_name}"

def get_authors(authors_list):
  if authors_list:
    names = [author['nome'] for author in authors_list]
    names.sort(key=lambda n: n.lower())
    normalized_names = [normalize_author_name(name) for name in names]
    normalized_names.sort()

    return (', '.join(names), ' '.join(normalized_names))

  return ("", "")

def extract_data(document, type, id_cnpq, name):
    dados = document[0].attrib
    detalhamento = document[1].attrib
    authors_list = [
      {
        'nome': el.attrib.get('NOME-COMPLETO-DO-AUTOR'),
        'nome_citacao': el.attrib.get('NOME-PARA-CITACAO'),
        'id_cnpq': el.attrib.get('NRO-ID-CNPQ') if el.attrib.get('NRO-ID-CNPQ') else ''
      }
      for el in document if el.tag == 'AUTORES'
    ]
    authors, normalized_authors = get_authors(authors_list)

    kind, field_type, country, homepage = specific_fields(type)
    
    normalizer = Normalizer()

    titulo = dados.get(f"TITULO{field_type}")
    titulo_norm_1 = normalizer.normalize(titulo)
    titulo_norm_2 = normalizer.remove_stopwords(titulo_norm_1)
    
    return {
      'lattes_origem': id_cnpq,
      'nome_origem': name,  
      'titulo': titulo,
      'titulo_ing': dados.get(f"TITULO{field_type}-INGLES"),
      'titulo_normalizado_1':  titulo_norm_1,
      'titulo_normalizado_2': titulo_norm_2,
      'categoria': kind,
      'subcategoria': type,
      'natureza': dados.get("NATUREZA"),
      'ano': dados.get(f"ANO{field_type}"),
      'pais': dados.get(f"PAIS-{country}"),
      'idioma': dados.get("IDIOMA"),
      'homepage': dados.get(f"HOME-PAGE{homepage}"),
      'doi': dados.get("DOI"),
      'periodico': detalhamento.get("TITULO-DO-PERIODICO-OU-REVISTA"),
      'evento': detalhamento.get("NOME-DO-EVENTO"),
      'cidade_evento': detalhamento.get("CIDADE-DO-EVENTO"),
      'ano_evento': detalhamento.get("ANO-DE-REALIZACAO"),
      'proceedings': detalhamento.get("TITULO-DOS-ANAIS-OU-PROCEEDINGS"),
      'isbn': detalhamento.get("ISBN"),
      'editora': detalhamento.get("NOME-DA-EDITORA"),
      'cidade_editora': detalhamento.get("CIDADE-DA-EDITORA"),
      'issn': detalhamento.get("ISSN"),
      'volume': detalhamento.get("VOLUME"),
      'fasciculo': detalhamento.get("FASCICULO"),
      'serie': detalhamento.get("SERIE"),
      'pag_inicial': detalhamento.get("PAGINA-INICIAL"),
      'pag_final': detalhamento.get("PAGINA-FINAL"),
      'local_publicacao': detalhamento.get("LOCAL-DE-PUBLICACAO"),
      'lista_autores': authors_list,
      'autores': authors,
      'autores_normalizados': normalized_authors,
      'tipo': dados.get("TIPO"),
      'tipo_ing': dados.get("TIPO-INGLES"),  
      'tipo_orientacao': detalhamento.get("TIPO-DE-ORIENTACAO-CONCLUIDA"),
      'cod_ies': detalhamento.get("CODIGO-INSTITUICAO"),
      'ies': detalhamento.get("NOME-DA-INSTITUICAO"),
      'cod_orgao': detalhamento.get("CODIGO-ORGAO"),
      'orgao': detalhamento.get("NOME-ORGAO"),
      'cod_curso': detalhamento.get("CODIGO-CURSO"),
      'curso': detalhamento.get("NOME-DO-CURSO"),
      'curso_ing': detalhamento.get("NOME-DO-CURSO-INGLES"),
      'bolsa': detalhamento.get("FLAG-BOLSA"),
      'cod_financiadora': detalhamento.get("CODIGO-AGENCIA-FINANCIADORA"),
      'financiadora': detalhamento.get("NOME-DA-AGENCIA"),
      'paginas': detalhamento.get("NUMERO-DE-PAGINAS"),
      'estudante': detalhamento.get("NOME-DO-ORIENTADO"),
      'id_estudante': detalhamento.get("NUMERO-ID-ORIENTADO"),
      'orientador': name,
      'id_orientador': id_cnpq
    }

def extract_collection(d, category, id, name):
  data = extract_data(d, category, id, name)
  # coauthors = [autor['id_cnpq'] for autor in data.get('lista_autores') if autor['id_cnpq']]
  return data

def import_from_xml(file, original=False):
  tree = ET.parse(file)
  root = tree.getroot()

  id = file.split("/")[-1].split(".xml")[0]
  name = root[0].attrib.get('NOME-COMPLETO')

  bibliografica = root[1] if len(root) > 1 else []

  artigos_em_eventos = [el for el in bibliografica if el.tag == 'TRABALHOS-EM-EVENTOS']
  artigos_em_eventos = artigos_em_eventos[0] if len(artigos_em_eventos) else []
  artigos_em_periodicos = [el for el in bibliografica if el.tag == 'ARTIGOS-PUBLICADOS']
  artigos_em_periodicos = artigos_em_periodicos[0] if len(artigos_em_periodicos) else []

  outra_producao = root[3] if len(root) > 3 else []
  outra_producao = [el for el in outra_producao if el.tag == 'ORIENTACOES-CONCLUIDAS']
  outra_producao = outra_producao[0] if len(outra_producao) else []
  orientacoes = [
    el for el in outra_producao
    if el.tag in [
      'ORIENTACOES-CONCLUIDAS-PARA-MESTRADO',
      'ORIENTACOES-CONCLUIDAS-PARA-DOUTORADO',
    ]
  ]
  tccs = [el for el in outra_producao if el[0].attrib.get('NATUREZA') == 'TRABALHO_DE_CONCLUSAO_DE_CURSO_GRADUACAO']

  collections = [
    ('artigo_em_evento', artigos_em_eventos),
    ('artigo_em_periodico', artigos_em_periodicos),
    ('tese_ou_dissertacao', orientacoes),
    ('tcc', tccs)
  ]
  productions = []
  coauthors = []
                       
  for category, collection in collections:
    if mode == 3:
      with Pool(nprocesses) as p:
        results = p.map(partial(extract_collection, category=category, id=id, name=name), collection)  
        coauthors += [autor['id_cnpq'] for data in results for autor in data.get('lista_autores') if autor['id_cnpq']]
        productions += results
    else:
      for d in collection:
        data = extract_data(d, category, id, name)
        coauthors += [autor['id_cnpq'] for autor in data.get('lista_autores') if autor['id_cnpq']]
        productions.append(data)
    
  while len(coauthors) > 20:
    coauthors.pop()

  coauthors.sort()
                         
  return productions, set(coauthors) - {id}

def create_folder(id):
    folder = f"{os.getcwd()}/{id}"

    if os.path.isdir(folder):
        i = 1
        new_folder = f"{folder} ({i})"
        while os.path.isdir(new_folder):
            i += 1
            new_folder = f"{folder} ({i})"

        folder = new_folder
    
    folder += "/"
    os.mkdir(folder)
    
    return folder

def delete_folder(folder):
    shutil.rmtree(folder)
    
def download_xml(id, folder):
    if os.path.isfile(f"curriculos/{id}.zip") and not os.path.isfile(f"{folder}{id}.xml"):
        with zipfile.ZipFile(f"curriculos/{id}.zip") as zip:
          fileinfo = zip.infolist()[0]
          fileinfo.filename = f"{id}.xml"
          zip.extract(fileinfo, f"{folder}")

        return True
    
    return False

def importar(id, coauthors, productions, folder):
  if download_xml(id, folder):
    new_productions, new_coauthors = import_from_xml(f"{folder}{id}.xml")

    return new_productions, new_coauthors
  
  return [], set()

def import_coauthors(coauthors, productions, folder):
    if mode == 4:
      with Pool(nprocesses) as p:
        results = p.map(partial(importar, coauthors=coauthors, productions=productions, folder=folder), coauthors)
        productions += [data for r in results for data in r[0]]
        coauthors = reduce(lambda s1, s2: s1.union(s2), [r[1] for r in results])

    else: 
      for id in coauthors:
        if download_xml(id, folder):
          new_productions, new_coauthors = import_from_xml(f"{folder}{id}.xml")
          productions += new_productions
          coauthors = coauthors.union(new_coauthors)

    return coauthors

def comparable_productions(left, right, fields):
    doi1, doi2 = left['doi'], right['doi']
    title1, title2 = left['titulo_normalizado_2'], right['titulo_normalizado_2']
    length_distance = len(title1) - len(title2)
    
    if (doi1 == '' or doi2  == '' or (fuzz.token_set_ratio(doi1, doi2) > 90)):
      for field in fields:
        if left[field] != right[field]:
            return True
        
    return False

def compare(right, left):
  if comparable_productions(left, right, fields):
    features = torch.FloatTensor(mlp.get_features(left, right))
    result = model(features)
    if result[0] < result[1]:
      return {
        'info': right,
        'different_fields': [field for field in fields if left[field] != right[field]]
      }
    
  return None

def compare_production(left):
    equal = 0
    matches = []
    
    if mode == 2:
      with Pool(nprocesses) as p:
        matches = p.map(partial(compare, left=left), productions)
        matches = [match for match in matches if match]

    else:
      for right in productions:
        match = compare(right, left)
        if match:
          matches.append(match)

    if len(matches) > 0:
        category = categoryNames[categories.index(left['subcategoria'])]
        
        return (category, {
            'original': {'info': left},
            'matches': matches
        })

    return (None, None)

def process_lattes(id):
  global productions, categories, categoryNames, fields, model
  
  categories = ['artigo_em_evento', 'artigo_em_periodico', 'tese_ou_dissertacao', 'tcc']
  categoryNames = ['Artigos em Eventos', 'Artigos em Periódicos', 'Orientações de Mestrado e Doutorado', 'Orientações de TCC']

  folder = create_folder(id)

  if not download_xml(id, folder):
    delete_folder(id)
    return None

  productions, coauthors = import_from_xml(f"{folder}{id}.xml", original=True)
  original_productions = [p for p in productions]

  response = {
      "origem": {
        "nome": original_productions[0].get('nome_origem'),
        "id": id 
      }, 
      categoryNames[0]: [],
      categoryNames[1]: [],
      categoryNames[2]: [],
      categoryNames[3]: []
  }

  # 1 Nivel de coautoria
  coauthors = import_coauthors(coauthors, productions, folder) - coauthors
  # print([coauthor for coauthor in coauthors if not os.path.exists(f"curriculos/{coauthor}.zip")])
  # 2 Niveis de coautoria
  import_coauthors(coauthors, productions, folder)

  delete_folder(id)
  print(f"Número de produções: {len(productions)}")

  model = mlp.MLP()
  model.load_state_dict(torch.load('./production_matcher', map_location=torch.device('cpu')))
  model.eval()

  exception_fields = ['lattes_origem', 'nome_origem', 'titulo_normalizado_1', 'titulo_normalizado_2']
  fields = [key for key in original_productions[0].keys() if key not in exception_fields]

  if mode == 1:
    with Pool(nprocesses) as p:
      productions = p.map(compare_production, original_productions)
      for category in categoryNames:
        categoryProductions = filter(lambda p: p[0] == category, productions)
        response[category] = list(map(lambda p: p[1], categoryProductions))

  else:
    for left in original_productions:
        category, result = compare_production(left)
        if category:
          response[category].append(result)

  return response

if __name__ == '__main__':
  id = sys.argv[1]
  mode = int(sys.argv[2])
  nprocesses = int(sys.argv[3])
  
  modes = [
    "Serial",
    "Paralelo em loop de comparação externo",
    "Paralelo em loop de comparação interno",
    "Paralelo em importação de xml",
    "Paralelo em importação de coautores"
  ]
  
  print(f"Modo {modes[mode]}")
  start_time = time.time()
  response = process_lattes(id)
  elapsed_time = time.time() - start_time
  print(elapsed_time)

  with open(f"result_{mode}", "w") as f:
    f.write(str(elapsed_time))